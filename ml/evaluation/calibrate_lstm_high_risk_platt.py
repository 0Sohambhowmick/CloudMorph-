
from pathlib import Path
import json
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


# =========================================================
# CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parents[2]

RAW_PATH = ROOT / "datasets" / "processed" / "server_usage_clean.csv"

LABEL_PATH = ROOT / "ml" / "data" / "optimization_training_dataset.csv"

MODEL_DIR = ROOT / "ml" / "models" / "lstm_sequence_classifier"

CHECKPOINT_PATH = MODEL_DIR / "lstm_workload_classifier.pt"

SCALER_PATH = MODEL_DIR / "feature_scaler.npz"

OUTPUT_DIR = ROOT / "ml" / "models" / "lstm_high_risk_platt"

FEATURES = [
    "util_CPU",
    "util_memory",
    "util_disk",
    "load1",
    "load5",
    "load15",
]

CLASS_NAMES = [
    "LOW_DEMAND",
    "NORMAL_DEMAND",
    "HIGH_DEMAND",
]

CLASS_TO_ID = {
    name: index for index, name in enumerate(CLASS_NAMES)
}

HIGH_ID = CLASS_TO_ID["HIGH_DEMAND"]

HISTORY_WINDOW = 12
FORECAST_HORIZON = 6

BATCH_SIZE = 256

ORIGINAL_RISK_THRESHOLD = 0.11
MIN_VALIDATION_PRECISION = 0.60

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# =========================================================
# ORIGINAL LSTM ARCHITECTURE
# =========================================================

class WorkloadLSTM(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        num_classes,
        dropout,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        output, _ = self.lstm(x)

        last_output = output[:, -1, :]

        return self.classifier(
            self.dropout(last_output)
        )


# =========================================================
# LOAD ORIGINAL MODEL AND SCALER
# =========================================================

def load_model():

    logger.info("Loading original LSTM checkpoint...")

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(CHECKPOINT_PATH)

    if not SCALER_PATH.exists():
        raise FileNotFoundError(SCALER_PATH)

    try:
        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=DEVICE,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=DEVICE,
        )

    if checkpoint["input_features"] != FEATURES:
        raise ValueError("Checkpoint feature order mismatch.")

    if checkpoint["class_names"] != CLASS_NAMES:
        raise ValueError("Checkpoint class order mismatch.")

    model = WorkloadLSTM(
        input_size=len(checkpoint["input_features"]),
        hidden_size=checkpoint["hidden_size"],
        num_layers=checkpoint["num_layers"],
        num_classes=len(checkpoint["class_names"]),
        dropout=checkpoint["dropout"],
    ).to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    scaler_data = np.load(SCALER_PATH)

    scaler_mean = scaler_data["mean"]
    scaler_scale = scaler_data["scale"]

    if len(scaler_mean) != len(FEATURES):
        raise ValueError("Scaler feature count mismatch.")

    if not np.isfinite(scaler_mean).all():
        raise ValueError("Invalid scaler mean.")

    if not np.isfinite(scaler_scale).all():
        raise ValueError("Invalid scaler scale.")

    if (scaler_scale <= 0).any():
        raise ValueError("Invalid scaler scale values.")

    logger.info(
        "Original LSTM loaded. Best epoch: %s",
        checkpoint.get("best_epoch", "Unknown"),
    )

    return model, scaler_mean, scaler_scale


# =========================================================
# LOAD DATA
# =========================================================

def load_data():

    logger.info("Loading raw data and labels...")

    raw = pd.read_csv(RAW_PATH)
    labels = pd.read_csv(LABEL_PATH)

    raw_required = {
        "machineID",
        "step",
        *FEATURES,
    }

    label_required = {
        "machineID",
        "history_start_step",
        "history_end_step",
        "target_end_step",
        "target",
    }

    if not raw_required.issubset(raw.columns):
        raise ValueError(
            f"Missing raw columns: {raw_required - set(raw.columns)}"
        )

    if not label_required.issubset(labels.columns):
        raise ValueError(
            f"Missing label columns: {label_required - set(labels.columns)}"
        )

    raw["machineID"] = raw["machineID"].astype(str)
    labels["machineID"] = labels["machineID"].astype(str)

    raw = raw.sort_values(
        ["machineID", "step"]
    ).reset_index(drop=True)

    labels["history_start_step"] = (
        labels["history_start_step"].astype(int)
    )

    labels["history_end_step"] = (
        labels["history_end_step"].astype(int)
    )

    labels["target_end_step"] = (
        labels["target_end_step"].astype(int)
    )

    labels = labels[
        labels["target"].isin(CLASS_TO_ID)
    ].copy()

    return raw, labels


# =========================================================
# BUILD SEQUENCES
# =========================================================

def build_sequences(raw, labels):

    logger.info("Building historical sequences...")

    machine_groups = {
        machine_id: group.set_index("step")
        for machine_id, group in raw.groupby("machineID")
    }

    sequences = []
    targets = []
    metadata = []

    for row in labels.itertuples(index=False):

        machine_id = str(row.machineID)
        start = int(row.history_start_step)

        if machine_id not in machine_groups:
            continue

        machine = machine_groups[machine_id]

        history_steps = list(
            range(start, start + HISTORY_WINDOW)
        )

        if not all(
            step in machine.index
            for step in history_steps
        ):
            continue

        history = machine.loc[
            history_steps,
            FEATURES,
        ]

        values = history.to_numpy(
            dtype=np.float32
        )

        if values.shape != (
            HISTORY_WINDOW,
            len(FEATURES),
        ):
            continue

        if not np.isfinite(values).all():
            continue

        sequences.append(values)

        targets.append(
            CLASS_TO_ID[row.target]
        )

        metadata.append({
            "machineID": machine_id,
            "history_start_step": start,
            "history_end_step": int(row.history_end_step),
            "target_end_step": int(row.target_end_step),
            "target": row.target,
        })

    X = np.asarray(
        sequences,
        dtype=np.float32,
    )

    y = np.asarray(
        targets,
        dtype=np.int64,
    )

    meta = pd.DataFrame(metadata)

    if len(X) == 0:
        raise ValueError("No valid sequences found.")

    logger.info(
        "Sequence shape: %s",
        X.shape,
    )

    return X, y, meta


# =========================================================
# CHRONOLOGICAL SPLIT
# =========================================================

def chronological_split(X, y, meta):

    train_mask = (
        meta["target_end_step"] < 51
    ).to_numpy()

    validation_mask = (
        (meta["history_start_step"] >= 51)
        &
        (meta["target_end_step"] < 89)
    ).to_numpy()

    test_mask = (
        meta["history_start_step"] >= 89
    ).to_numpy()

    X_train = X[train_mask]
    X_val = X[validation_mask]
    X_test = X[test_mask]

    y_val = y[validation_mask]
    y_test = y[test_mask]

    meta_val = meta.loc[
        validation_mask
    ].reset_index(drop=True)

    meta_test = meta.loc[
        test_mask
    ].reset_index(drop=True)

    expected = (
        33150,
        20475,
        37050,
    )

    actual = (
        int(train_mask.sum()),
        len(X_val),
        len(X_test),
    )

    if actual != expected:
        raise ValueError(
            f"Unexpected split sizes. "
            f"Expected {expected}, got {actual}."
        )

    logger.info(
        "TRAIN: %s | VALIDATION: %s | TEST: %s",
        actual[0],
        len(X_val),
        len(X_test),
    )

    logger.info(
        "PURGED WINDOWS: %s",
        len(X) - sum(actual),
    )

    return (
        X_val,
        y_val,
        meta_val,
        X_test,
        y_test,
        meta_test,
    )


# =========================================================
# APPLY SAVED SCALER
# =========================================================

def normalize_sequences(X_val, X_test, mean, scale):

    X_val = (
        X_val - mean
    ) / scale

    X_test = (
        X_test - mean
    ) / scale

    return (
        X_val.astype(np.float32),
        X_test.astype(np.float32),
    )


# =========================================================
# GET MODEL LOGITS
# =========================================================

def get_logits(model, X):

    model.eval()

    all_logits = []

    with torch.no_grad():

        for start in range(0, len(X), BATCH_SIZE):

            batch = torch.tensor(
                X[start:start + BATCH_SIZE],
                dtype=torch.float32,
                device=DEVICE,
            )

            logits = model(batch)

            all_logits.append(
                logits.cpu()
            )

    return torch.cat(
        all_logits,
        dim=0,
    ).numpy()


# =========================================================
# CONVERT MULTICLASS LOGITS TO BINARY HIGH-RISK LOGITS
# =========================================================

def binary_high_risk_logit(multiclass_logits):

    logits = multiclass_logits.astype(np.float64)

    high_logits = logits[:, HIGH_ID]

    other_logits = np.delete(
        logits,
        HIGH_ID,
        axis=1,
    )

    # log(P(HIGH) / (1 - P(HIGH)))
    # This is mathematically equivalent to:
    # HIGH logit - logsumexp(LOW logit, NORMAL logit).

    maximum = np.max(
        other_logits,
        axis=1,
        keepdims=True,
    )

    logsumexp_other = (
        maximum[:, 0]
        +
        np.log(
            np.exp(other_logits - maximum).sum(axis=1)
        )
    )

    return high_logits - logsumexp_other


# =========================================================
# FIT PLATT SCALING
# Validation data only.
# =========================================================

def fit_platt_scaler(val_binary_logits, y_val):

    logger.info(
        "Fitting binary Platt scaling on validation data only..."
    )

    y_binary = (
        y_val == HIGH_ID
    ).astype(int)

    if len(np.unique(y_binary)) != 2:
        raise ValueError(
            "Validation data must contain both binary classes."
        )

    X_calibration = val_binary_logits.reshape(
        -1, 1
    )

    # Unregularized logistic calibration.
    # This is preferable here to arbitrary class weighting:
    # the goal is probability calibration, not class rebalancing.
    calibrator = LogisticRegression(
        C=1e6,
        solver="lbfgs",
        max_iter=2000,
        random_state=42,
    )

    calibrator.fit(
        X_calibration,
        y_binary,
    )

    slope = float(
        calibrator.coef_[0, 0]
    )

    intercept = float(
        calibrator.intercept_[0]
    )

    if slope <= 0:
        raise RuntimeError(
            "Platt scaling produced a non-positive slope. "
            "Do not use this calibration model."
        )

    logger.info(
        "Platt slope: %.8f | Intercept: %.8f",
        slope,
        intercept,
    )

    return calibrator, slope, intercept


# =========================================================
# CALIBRATED HIGH-RISK PROBABILITIES
# =========================================================

def calibrated_probabilities(
    calibrator,
    binary_logits,
):

    return calibrator.predict_proba(
        binary_logits.reshape(-1, 1)
    )[:, 1]


# =========================================================
# THRESHOLD SELECTION
# Validation data only.
# =========================================================

def select_risk_threshold(y_val, val_probs):

    actual_high = (
        y_val == HIGH_ID
    ).astype(int)

    candidates = np.unique(
        np.concatenate([
            np.arange(0.01, 0.951, 0.01),
            val_probs,
        ])
    )

    eligible = []

    for threshold in np.sort(candidates):

        alerts = (
            val_probs >= threshold
        ).astype(int)

        precision = precision_score(
            actual_high,
            alerts,
            zero_division=0,
        )

        recall = recall_score(
            actual_high,
            alerts,
            zero_division=0,
        )

        f1 = f1_score(
            actual_high,
            alerts,
            zero_division=0,
        )

        if precision >= MIN_VALIDATION_PRECISION:

            eligible.append({
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "alerts": int(alerts.sum()),
            })

    if not eligible:
        raise RuntimeError(
            "No threshold achieved minimum validation precision."
        )

    best = max(
        eligible,
        key=lambda item: (
            item["f1"],
            item["precision"],
        ),
    )

    return best, pd.DataFrame(eligible)


# =========================================================
# CALIBRATION METRICS
# =========================================================

def calculate_calibration_metrics(y, probs):

    actual_high = (
        y == HIGH_ID
    ).astype(int)

    probs = np.clip(
        probs,
        1e-12,
        1.0 - 1e-12,
    )

    return {
        "binary_log_loss": float(
            log_loss(
                actual_high,
                probs,
                labels=[0, 1],
            )
        ),
        "binary_brier_score": float(
            brier_score_loss(
                actual_high,
                probs,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                actual_high,
                probs,
            )
        ),
        "mean_high_probability": float(
            probs.mean()
        ),
        "median_high_probability": float(
            np.median(probs)
        ),
        "actual_high_prevalence": float(
            actual_high.mean()
        ),
    }


# =========================================================
# EXPECTED CALIBRATION ERROR
# =========================================================

def calculate_ece(y, probs, bins=15):

    actual_high = (
        y == HIGH_ID
    ).astype(int)

    edges = np.linspace(
        0.0,
        1.0,
        bins + 1,
    )

    ece = 0.0
    records = []

    for index in range(bins):

        lower = edges[index]
        upper = edges[index + 1]

        if index == bins - 1:

            mask = (
                (probs >= lower)
                &
                (probs <= upper)
            )

        else:

            mask = (
                (probs >= lower)
                &
                (probs < upper)
            )

        count = int(mask.sum())

        if count == 0:
            continue

        mean_probability = float(
            probs[mask].mean()
        )

        observed_rate = float(
            actual_high[mask].mean()
        )

        contribution = (
            count / len(probs)
        ) * abs(
            mean_probability - observed_rate
        )

        ece += contribution

        records.append({
            "bin_lower": float(lower),
            "bin_upper": float(upper),
            "count": count,
            "mean_predicted_probability": mean_probability,
            "observed_high_rate": observed_rate,
            "ece_contribution": float(contribution),
        })

    return float(ece), pd.DataFrame(records)


# =========================================================
# RISK EVALUATION
# =========================================================

def evaluate_risk(y, probs, threshold):

    actual_high = (
        y == HIGH_ID
    ).astype(int)

    alerts = (
        probs >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        actual_high,
        alerts,
        labels=[0, 1],
    ).ravel()

    precision = precision_score(
        actual_high,
        alerts,
        zero_division=0,
    )

    recall = recall_score(
        actual_high,
        alerts,
        zero_division=0,
    )

    f1 = f1_score(
        actual_high,
        alerts,
        zero_division=0,
    )

    specificity = (
        tn / (tn + fp)
        if tn + fp else 0.0
    )

    return {
        "threshold": float(threshold),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
        "alerts": int(alerts.sum()),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": float(specificity),
    }


# =========================================================
# MAIN
# =========================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger.info("Device: %s", DEVICE)

    # Load the original model and its saved scaler.
    model, scaler_mean, scaler_scale = load_model()

    # Rebuild exact sequences and chronological splits.
    raw, labels = load_data()

    X, y, metadata = build_sequences(
        raw,
        labels,
    )

    (
        X_val,
        y_val,
        meta_val,
        X_test,
        y_test,
        meta_test,
    ) = chronological_split(
        X,
        y,
        metadata,
    )

    X_val, X_test = normalize_sequences(
        X_val,
        X_test,
        scaler_mean,
        scaler_scale,
    )

    # Obtain unchanged model logits.
    logger.info("Generating validation logits...")

    val_logits = get_logits(
        model,
        X_val,
    )

    logger.info("Generating test logits...")

    test_logits = get_logits(
        model,
        X_test,
    )

    # Convert to binary HIGH_DEMAND logits.
    val_binary_logits = binary_high_risk_logit(
        val_logits
    )

    test_binary_logits = binary_high_risk_logit(
        test_logits
    )

    # Original probabilities.
    val_original_probs = 1.0 / (
        1.0 + np.exp(
            -np.clip(val_binary_logits, -500, 500)
        )
    )

    test_original_probs = 1.0 / (
        1.0 + np.exp(
            -np.clip(test_binary_logits, -500, 500)
        )
    )

    # Fit Platt calibration using validation only.
    calibrator, slope, intercept = fit_platt_scaler(
        val_binary_logits,
        y_val,
    )

    val_platt_probs = calibrated_probabilities(
        calibrator,
        val_binary_logits,
    )

    test_platt_probs = calibrated_probabilities(
        calibrator,
        test_binary_logits,
    )

    # Select calibrated threshold on validation only.
    best_threshold, threshold_table = (
        select_risk_threshold(
            y_val,
            val_platt_probs,
        )
    )

    logger.info(
        "Selected Platt validation threshold: %s",
        best_threshold,
    )

    # Evaluate original model at original threshold.
    original_test_metrics = evaluate_risk(
        y_test,
        test_original_probs,
        ORIGINAL_RISK_THRESHOLD,
    )

    # Evaluate calibrated probabilities at the same original threshold.
    platt_fixed_threshold_metrics = evaluate_risk(
        y_test,
        test_platt_probs,
        ORIGINAL_RISK_THRESHOLD,
    )

    # Evaluate calibrated probabilities at the validation-selected threshold.
    platt_selected_threshold_metrics = evaluate_risk(
        y_test,
        test_platt_probs,
        best_threshold["threshold"],
    )

    # Probability calibration metrics.
    original_validation_calibration = (
        calculate_calibration_metrics(
            y_val,
            val_original_probs,
        )
    )

    platt_validation_calibration = (
        calculate_calibration_metrics(
            y_val,
            val_platt_probs,
        )
    )

    original_test_calibration = (
        calculate_calibration_metrics(
            y_test,
            test_original_probs,
        )
    )

    platt_test_calibration = (
        calculate_calibration_metrics(
            y_test,
            test_platt_probs,
        )
    )

    original_val_ece, _ = calculate_ece(
        y_val,
        val_original_probs,
    )

    platt_val_ece, val_reliability = calculate_ece(
        y_val,
        val_platt_probs,
    )

    original_test_ece, _ = calculate_ece(
        y_test,
        test_original_probs,
    )

    platt_test_ece, test_reliability = calculate_ece(
        y_test,
        test_platt_probs,
    )

    original_validation_calibration["ece"] = original_val_ece
    platt_validation_calibration["ece"] = platt_val_ece

    original_test_calibration["ece"] = original_test_ece
    platt_test_calibration["ece"] = platt_test_ece

    # Save test predictions.
    predictions = meta_test.copy()

    predictions["original_high_probability"] = (
        test_original_probs
    )

    predictions["platt_high_probability"] = (
        test_platt_probs
    )

    predictions["original_risk_alert"] = np.where(
        test_original_probs >= ORIGINAL_RISK_THRESHOLD,
        "HIGH_RISK",
        "NO_HIGH_RISK",
    )

    predictions["platt_fixed_threshold_alert"] = np.where(
        test_platt_probs >= ORIGINAL_RISK_THRESHOLD,
        "HIGH_RISK",
        "NO_HIGH_RISK",
    )

    predictions["platt_selected_threshold_alert"] = np.where(
        test_platt_probs >= best_threshold["threshold"],
        "HIGH_RISK",
        "NO_HIGH_RISK",
    )

    predictions.to_csv(
        OUTPUT_DIR / "test_predictions_platt.csv",
        index=False,
    )

    # Save threshold search.
    threshold_table.to_csv(
        OUTPUT_DIR / "validation_threshold_analysis.csv",
        index=False,
    )

    # Save reliability bins.
    val_reliability.to_csv(
        OUTPUT_DIR / "validation_calibration_reliability.csv",
        index=False,
    )

    test_reliability.to_csv(
        OUTPUT_DIR / "test_calibration_reliability.csv",
        index=False,
    )

    # Save validation calibration metrics.
    with open(
        OUTPUT_DIR / "validation_calibration_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            {
                "original": original_validation_calibration,
                "platt_calibrated": platt_validation_calibration,
            },
            file,
            indent=4,
        )

    # Save test calibration metrics.
    with open(
        OUTPUT_DIR / "test_calibration_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            {
                "original": original_test_calibration,
                "platt_calibrated": platt_test_calibration,
            },
            file,
            indent=4,
        )

    # Save complete report.
    summary = {
        "experiment": "Binary HIGH_DEMAND Platt Scaling",
        "calibration_method": "Logistic regression on binary high-risk logit",
        "training_performed": False,
        "checkpoint": str(CHECKPOINT_PATH),
        "validation_samples": len(y_val),
        "test_samples": len(y_test),
        "platt_slope": slope,
        "platt_intercept": intercept,
        "validation_selected_threshold": best_threshold,
        "original_threshold": ORIGINAL_RISK_THRESHOLD,
        "validation_calibration": {
            "original": original_validation_calibration,
            "platt_calibrated": platt_validation_calibration,
        },
        "test_calibration": {
            "original": original_test_calibration,
            "platt_calibrated": platt_test_calibration,
        },
        "test_risk_metrics": {
            "original_lstm_original_threshold": original_test_metrics,
            "platt_original_threshold": platt_fixed_threshold_metrics,
            "platt_validation_selected_threshold": platt_selected_threshold_metrics,
        },
    }

    with open(
        OUTPUT_DIR / "platt_calibration_report.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    # Console output.
    print("\n" + "=" * 70)
    print("HIGH-DEMAND PLATT SCALING RESULTS")
    print("=" * 70)

    print(f"\nPlatt slope: {slope:.8f}")
    print(f"Platt intercept: {intercept:.8f}")

    print("\nSELECTED VALIDATION THRESHOLD")
    print(
        json.dumps(
            best_threshold,
            indent=4,
        )
    )

    print("\nTEST CALIBRATION METRICS")

    print(
        pd.DataFrame({
            "Original": original_test_calibration,
            "Platt calibrated": platt_test_calibration,
        }).to_string()
    )

    print("\nTEST RISK METRICS")

    print(
        pd.DataFrame({
            "Original LSTM": original_test_metrics,
            "Platt at original threshold": platt_fixed_threshold_metrics,
            "Platt at selected threshold": platt_selected_threshold_metrics,
        }).to_string()
    )

    print("\n" + "=" * 70)
    print("PLATT CALIBRATION COMPLETED")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
