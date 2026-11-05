
from pathlib import Path
import json
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


# =========================================================
# CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parents[2]

RAW_PATH = ROOT / "datasets" / "processed" / "server_usage_clean.csv"

LABEL_PATH = ROOT / "ml" / "data" / "optimization_training_dataset.csv"

# Original model: read-only.
MODEL_DIR = ROOT / "ml" / "models" / "lstm_sequence_classifier"

CHECKPOINT_PATH = MODEL_DIR / "lstm_workload_classifier.pt"
SCALER_PATH = MODEL_DIR / "feature_scaler.npz"

# All calibration results go into a separate directory.
OUTPUT_DIR = ROOT / "ml" / "models" / "lstm_temperature_calibration"

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
# MODEL ARCHITECTURE
# Must match the original LSTM exactly.
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
# LOAD ORIGINAL CHECKPOINT
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

    model = WorkloadLSTM(
        input_size=len(checkpoint["input_features"]),
        hidden_size=checkpoint["hidden_size"],
        num_layers=checkpoint["num_layers"],
        num_classes=len(checkpoint["class_names"]),
        dropout=checkpoint["dropout"],
    ).to(DEVICE)

    if checkpoint["input_features"] != FEATURES:
        raise ValueError("Checkpoint feature order does not match.")

    if checkpoint["class_names"] != CLASS_NAMES:
        raise ValueError("Checkpoint class order does not match.")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    scaler_data = np.load(SCALER_PATH)

    scaler_mean = scaler_data["mean"]
    scaler_scale = scaler_data["scale"]

    if len(scaler_mean) != len(FEATURES):
        raise ValueError("Saved scaler feature count is incorrect.")

    if len(scaler_scale) != len(FEATURES):
        raise ValueError("Saved scaler scale count is incorrect.")

    if not np.isfinite(scaler_mean).all():
        raise ValueError("Scaler mean contains invalid values.")

    if not np.isfinite(scaler_scale).all():
        raise ValueError("Scaler scale contains invalid values.")

    if (scaler_scale <= 0).any():
        raise ValueError("Scaler contains zero or negative scales.")

    logger.info(
        "Original checkpoint loaded. Best epoch: %s",
        checkpoint.get("best_epoch", "Unknown"),
    )

    return model, scaler_mean, scaler_scale


# =========================================================
# LOAD DATA
# =========================================================

def load_data():
    logger.info("Loading raw workload data and labels...")

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
# BUILD HISTORICAL SEQUENCES
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
        targets.append(CLASS_TO_ID[row.target])

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
        raise ValueError("No valid sequences were created.")

    logger.info(
        "Sequence shape: %s",
        X.shape,
    )

    return X, y, meta


# =========================================================
# SAME CHRONOLOGICAL SPLIT
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

    X_train, y_train = X[train_mask], y[train_mask]

    X_val, y_val = X[validation_mask], y[validation_mask]

    X_test, y_test = X[test_mask], y[test_mask]

    meta_val = meta.loc[
        validation_mask
    ].reset_index(drop=True)

    meta_test = meta.loc[
        test_mask
    ].reset_index(drop=True)

    expected = (33150, 20475, 37050)

    actual = (
        len(X_train),
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
        len(X_train),
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
# APPLY SAVED TRAINING SCALER
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
    )


# =========================================================
# TEMPERATURE SCALING
# Fit ONLY on validation logits and labels.
# =========================================================

def fit_temperature(val_logits, y_val):

    logger.info("Fitting temperature using validation data only...")

    logits = val_logits.to(DEVICE)

    targets = torch.tensor(
        y_val,
        dtype=torch.long,
        device=DEVICE,
    )

    # Optimize log-temperature so temperature is always positive.
    log_temperature = nn.Parameter(
        torch.zeros(
            1,
            device=DEVICE,
        )
    )

    optimizer = torch.optim.LBFGS(
        [log_temperature],
        lr=0.05,
        max_iter=100,
        line_search_fn="strong_wolfe",
    )

    criterion = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()

        temperature = torch.exp(log_temperature)

        calibrated_logits = logits / temperature

        loss = criterion(
            calibrated_logits,
            targets,
        )

        loss.backward()

        return loss

    optimizer.step(closure)

    temperature = float(
        torch.exp(log_temperature).detach().cpu().item()
    )

    if not np.isfinite(temperature) or temperature <= 0:
        raise RuntimeError("Invalid fitted temperature.")

    logger.info(
        "Fitted temperature: %.6f",
        temperature,
    )

    return temperature


# =========================================================
# PROBABILITIES
# =========================================================

def probabilities_from_logits(logits, temperature=1.0):

    scaled_logits = logits / temperature

    return torch.softmax(
        scaled_logits,
        dim=1,
    ).cpu().numpy()


# =========================================================
# SELECT RISK THRESHOLD ON VALIDATION DATA
# =========================================================

def select_risk_threshold(y_val, val_probs):

    high_id = CLASS_TO_ID["HIGH_DEMAND"]

    actual_high = (
        y_val == high_id
    ).astype(int)

    high_probs = val_probs[:, high_id]

    candidates = np.unique(
        np.concatenate([
            np.arange(0.01, 0.951, 0.01),
            high_probs,
        ])
    )

    eligible = []

    for threshold in np.sort(candidates):

        alerts = (
            high_probs >= threshold
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
            "No validation threshold met minimum precision."
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

def expected_calibration_error(confidence, correct, bins=15):

    confidence = np.asarray(confidence)
    correct = np.asarray(correct).astype(float)

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
                (confidence >= lower)
                & (confidence <= upper)
            )
        else:
            mask = (
                (confidence >= lower)
                & (confidence < upper)
            )

        count = int(mask.sum())

        if count == 0:
            continue

        mean_confidence = float(
            confidence[mask].mean()
        )

        accuracy = float(
            correct[mask].mean()
        )

        contribution = (
            count / len(confidence)
        ) * abs(
            accuracy - mean_confidence
        )

        ece += contribution

        records.append({
            "bin_lower": float(lower),
            "bin_upper": float(upper),
            "count": count,
            "mean_confidence": mean_confidence,
            "observed_accuracy": accuracy,
            "ece_contribution": float(contribution),
        })

    return float(ece), pd.DataFrame(records)


def calibration_metrics(y, probs):

    predictions = probs.argmax(axis=1)

    confidence = probs.max(axis=1)

    correct = (
        predictions == y
    ).astype(int)

    eps = 1e-12

    clipped = np.clip(
        probs,
        eps,
        1.0 - eps,
    )

    one_hot = np.eye(
        len(CLASS_NAMES)
    )[y]

    nll = -np.log(
        clipped[np.arange(len(y)), y]
    ).mean()

    multiclass_brier = np.mean(
        np.sum(
            (probs - one_hot) ** 2,
            axis=1,
        )
    )

    multiclass_ece, reliability = (
        expected_calibration_error(
            confidence,
            correct,
        )
    )

    high_id = CLASS_TO_ID["HIGH_DEMAND"]

    actual_high = (
        y == high_id
    ).astype(float)

    high_probs = probs[:, high_id]

    high_brier = np.mean(
        (high_probs - actual_high) ** 2
    )

    high_ece, high_reliability = (
        expected_calibration_error(
            high_probs,
            actual_high,
        )
    )

    metrics = {
        "negative_log_likelihood": float(nll),
        "multiclass_brier_score": float(multiclass_brier),
        "multiclass_ece": float(multiclass_ece),
        "high_demand_brier_score": float(high_brier),
        "high_demand_ece": float(high_ece),
    }

    return (
        metrics,
        reliability,
        high_reliability,
    )


# =========================================================
# RISK AND CLASSIFICATION METRICS
# =========================================================

def evaluate_predictions(y, probs, threshold):

    predicted = probs.argmax(axis=1)

    high_id = CLASS_TO_ID["HIGH_DEMAND"]

    actual_high = (
        y == high_id
    ).astype(int)

    alerts = (
        probs[:, high_id] >= threshold
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

    report = {
        "classification_accuracy": float(
            accuracy_score(y, predicted)
        ),
        "classification_balanced_accuracy": float(
            balanced_accuracy_score(y, predicted)
        ),
        "classification_macro_f1": float(
            f1_score(
                y,
                predicted,
                average="macro",
                zero_division=0,
            )
        ),
        "risk_threshold": float(threshold),
        "risk_precision": float(precision),
        "risk_recall": float(recall),
        "risk_f1": float(f1),
        "risk_specificity": float(specificity),
        "risk_true_positives": int(tp),
        "risk_false_positives": int(fp),
        "risk_false_negatives": int(fn),
        "risk_true_negatives": int(tn),
        "risk_alert_count": int(alerts.sum()),
    }

    return report, predicted, alerts


# =========================================================
# MAIN
# =========================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger.info("Device: %s", DEVICE)

    # Load original trained model and saved scaler.
    model, scaler_mean, scaler_scale = load_model()

    # Reconstruct the exact historical sequences.
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

    # Obtain logits from the unchanged model.
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

    # Uncalibrated probabilities.
    val_probs_original = probabilities_from_logits(
        val_logits,
        temperature=1.0,
    )

    test_probs_original = probabilities_from_logits(
        test_logits,
        temperature=1.0,
    )

    # Fit temperature using validation only.
    temperature = fit_temperature(
        val_logits,
        y_val,
    )

    # Apply the same temperature to validation and test.
    val_probs_calibrated = probabilities_from_logits(
        val_logits,
        temperature=temperature,
    )

    test_probs_calibrated = probabilities_from_logits(
        test_logits,
        temperature=temperature,
    )

    # Select calibrated threshold using validation only.
    calibrated_threshold, threshold_table = (
        select_risk_threshold(
            y_val,
            val_probs_calibrated,
        )
    )

    logger.info(
        "Calibrated validation threshold: %s",
        calibrated_threshold,
    )

    # Original model with its original threshold.
    original_report, original_predicted, original_alerts = (
        evaluate_predictions(
            y_test,
            test_probs_original,
            ORIGINAL_RISK_THRESHOLD,
        )
    )

    # Calibrated model with the newly selected threshold.
    calibrated_report, calibrated_predicted, calibrated_alerts = (
        evaluate_predictions(
            y_test,
            test_probs_calibrated,
            calibrated_threshold["threshold"],
        )
    )

    # Also evaluate calibrated probabilities at the original
    # threshold to separate calibration effects from threshold effects.
    fixed_threshold_report, _, fixed_threshold_alerts = (
        evaluate_predictions(
            y_test,
            test_probs_calibrated,
            ORIGINAL_RISK_THRESHOLD,
        )
    )

    # Calibration metrics on validation and test.
    original_val_calibration, _, _ = calibration_metrics(
        y_val,
        val_probs_original,
    )

    calibrated_val_calibration, val_reliability, val_high_reliability = (
        calibration_metrics(
            y_val,
            val_probs_calibrated,
        )
    )

    original_test_calibration, _, _ = calibration_metrics(
        y_test,
        test_probs_original,
    )

    calibrated_test_calibration, test_reliability, test_high_reliability = (
        calibration_metrics(
            y_test,
            test_probs_calibrated,
        )
    )

    # Save test predictions.
    predictions = meta_test.copy()

    predictions["original_high_probability"] = (
        test_probs_original[:, CLASS_TO_ID["HIGH_DEMAND"]]
    )

    predictions["calibrated_high_probability"] = (
        test_probs_calibrated[:, CLASS_TO_ID["HIGH_DEMAND"]]
    )

    predictions["original_risk_alert"] = np.where(
        original_alerts == 1,
        "HIGH_RISK",
        "NO_HIGH_RISK",
    )

    predictions["calibrated_risk_alert"] = np.where(
        calibrated_alerts == 1,
        "HIGH_RISK",
        "NO_HIGH_RISK",
    )

    predictions["calibrated_fixed_threshold_alert"] = np.where(
        fixed_threshold_alerts == 1,
        "HIGH_RISK",
        "NO_HIGH_RISK",
    )

    predictions["original_predicted_workload"] = (
        np.array(CLASS_NAMES)[original_predicted]
    )

    predictions["calibrated_predicted_workload"] = (
        np.array(CLASS_NAMES)[calibrated_predicted]
    )

    for index, class_name in enumerate(CLASS_NAMES):

        predictions[
            f"original_probability_{class_name}"
        ] = test_probs_original[:, index]

        predictions[
            f"calibrated_probability_{class_name}"
        ] = test_probs_calibrated[:, index]

    predictions.to_csv(
        OUTPUT_DIR / "test_predictions_calibrated.csv",
        index=False,
    )

    # Save validation threshold search.
    threshold_table.to_csv(
        OUTPUT_DIR / "validation_threshold_analysis.csv",
        index=False,
    )

    # Save reliability tables.
    val_reliability.to_csv(
        OUTPUT_DIR / "validation_multiclass_reliability.csv",
        index=False,
    )

    val_high_reliability.to_csv(
        OUTPUT_DIR / "validation_high_demand_reliability.csv",
        index=False,
    )

    test_reliability.to_csv(
        OUTPUT_DIR / "test_multiclass_reliability.csv",
        index=False,
    )

    test_high_reliability.to_csv(
        OUTPUT_DIR / "test_high_demand_reliability.csv",
        index=False,
    )

    # Save classification report.
    pd.DataFrame(
        classification_report(
            y_test,
            calibrated_predicted,
            labels=[0, 1, 2],
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        )
    ).transpose().to_csv(
        OUTPUT_DIR / "calibrated_classification_report.csv"
    )

    # Summary report.
    summary = {
        "experiment": "Original LSTM Temperature Scaling",
        "temperature": temperature,
        "calibration_method": "Validation-only temperature scaling",
        "checkpoint_used": str(CHECKPOINT_PATH),
        "scaler_used": str(SCALER_PATH),
        "training_performed": False,
        "validation_samples": len(y_val),
        "test_samples": len(y_test),
        "original_threshold": ORIGINAL_RISK_THRESHOLD,
        "calibrated_validation_threshold": calibrated_threshold,
        "validation_calibration": {
            "original": original_val_calibration,
            "calibrated": calibrated_val_calibration,
        },
        "test_calibration": {
            "original": original_test_calibration,
            "calibrated": calibrated_test_calibration,
        },
        "test_risk_metrics": {
            "original_model_original_threshold": original_report,
            "calibrated_model_calibrated_threshold": calibrated_report,
            "calibrated_model_original_threshold": fixed_threshold_report,
        },
    }

    with open(
        OUTPUT_DIR / "temperature_calibration_report.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    # Console results.
    print("\n" + "=" * 70)
    print("LSTM TEMPERATURE CALIBRATION RESULTS")
    print("=" * 70)

    print(f"\nFitted temperature: {temperature:.6f}")

    print("\nVALIDATION CALIBRATION METRICS")

    print(
        pd.DataFrame({
            "Original": original_val_calibration,
            "Calibrated": calibrated_val_calibration,
        }).to_string()
    )

    print("\nTEST CALIBRATION METRICS")

    print(
        pd.DataFrame({
            "Original": original_test_calibration,
            "Calibrated": calibrated_test_calibration,
        }).to_string()
    )

    print("\nTEST RISK METRICS")

    print(
        pd.DataFrame({
            "Original LSTM": original_report,
            "Calibrated LSTM": calibrated_report,
            "Calibrated at original threshold": fixed_threshold_report,
        }).to_string()
    )

    print("\nCALIBRATED CLASSIFICATION REPORT")

    print(
        classification_report(
            y_test,
            calibrated_predicted,
            labels=[0, 1, 2],
            target_names=CLASS_NAMES,
            zero_division=0,
        )
    )

    print("\n" + "=" * 70)
    print("TEMPERATURE CALIBRATION COMPLETED")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
