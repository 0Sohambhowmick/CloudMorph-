
from pathlib import Path
import json
import logging
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
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

OUTPUT_DIR = (
    ROOT / "ml" / "models" / "lstm_sequence_classifier"
)

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

# Same chronological boundaries as the previous
# 40/30/30 temporal threshold experiment.
TRAIN_TARGET_END_MAX = 50
VALIDATION_HISTORY_START_MIN = 51
VALIDATION_TARGET_END_MAX = 88
TEST_HISTORY_START_MIN = 89

BATCH_SIZE = 256
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.20
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
MAX_EPOCHS = 40
PATIENCE = 7
MIN_VALIDATION_PRECISION = 0.60
SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# =========================================================
# REPRODUCIBILITY
# =========================================================

def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


# =========================================================
# LSTM MODEL
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

        # Use the final historical observation's hidden state.
        last_output = output[:, -1, :]

        return self.classifier(
            self.dropout(last_output)
        )


# =========================================================
# LOAD AND VALIDATE DATA
# =========================================================

def load_data():
    logger.info("Loading raw workload data and labels...")

    raw = pd.read_csv(RAW_PATH)
    labels = pd.read_csv(LABEL_PATH)

    raw_required = {
        "machineID", "step", *FEATURES
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

    labels = labels.copy()

    labels["history_start_step"] = (
        labels["history_start_step"].astype(int)
    )

    labels["target_end_step"] = (
        labels["target_end_step"].astype(int)
    )

    labels = labels[
        labels["target"].isin(CLASS_TO_ID)
    ].copy()

    return raw, labels


# =========================================================
# BUILD 12 x 6 SEQUENCES
# =========================================================

def build_sequences(raw, labels):
    logger.info("Building historical workload sequences...")

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

        if not all(step in machine.index for step in history_steps):
            continue

        history = machine.loc[history_steps, FEATURES]

        values = history.to_numpy(dtype=np.float32)

        if values.shape != (HISTORY_WINDOW, len(FEATURES)):
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

    X = np.asarray(sequences, dtype=np.float32)
    y = np.asarray(targets, dtype=np.int64)
    meta = pd.DataFrame(metadata)

    if len(X) == 0:
        raise ValueError("No valid sequences were created.")

    logger.info(
        "Sequence shape: %s | Labels: %s",
        X.shape,
        y.shape,
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

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[validation_mask], y[validation_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    meta_train = meta.loc[train_mask].reset_index(drop=True)
    meta_val = meta.loc[validation_mask].reset_index(drop=True)
    meta_test = meta.loc[test_mask].reset_index(drop=True)

    logger.info(
        "TRAIN: %s | VALIDATION: %s | TEST: %s",
        len(X_train),
        len(X_val),
        len(X_test),
    )

    logger.info(
        "PURGED WINDOWS: %s",
        len(X) - len(X_train) - len(X_val) - len(X_test),
    )

    expected = (33150, 20475, 37050)

    actual = (len(X_train), len(X_val), len(X_test))

    if actual != expected:
        raise ValueError(
            f"Unexpected split sizes. Expected {expected}, got {actual}. "
            "Check dataset alignment and chronological boundaries."
        )

    return (
        X_train, y_train, meta_train,
        X_val, y_val, meta_val,
        X_test, y_test, meta_test,
    )


# =========================================================
# NORMALIZATION — FIT TRAIN DATA ONLY
# =========================================================

def normalize_sequences(X_train, X_val, X_test):
    scaler = StandardScaler()

    scaler.fit(
        X_train.reshape(-1, X_train.shape[-1])
    )

    def transform(values):
        shape = values.shape

        scaled = scaler.transform(
            values.reshape(-1, shape[-1])
        )

        return scaled.reshape(shape).astype(np.float32)

    return (
        transform(X_train),
        transform(X_val),
        transform(X_test),
        scaler,
    )


# =========================================================
# DATALOADERS
# =========================================================

def create_loaders(X_train, y_train, X_val, y_val):
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )

    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader


# =========================================================
# PREDICTIONS
# =========================================================

def predict(model, X):
    model.eval()

    loader = DataLoader(
        TensorDataset(
            torch.tensor(X, dtype=torch.float32)
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    probabilities = []

    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(DEVICE)

            logits = model(batch)

            probs = torch.softmax(logits, dim=1)

            probabilities.append(
                probs.cpu().numpy()
            )

    return np.concatenate(probabilities, axis=0)


# =========================================================
# TRAINING
# =========================================================

def train_model(
    X_train,
    y_train,
    X_val,
    y_val,
):
    train_loader, val_loader = create_loaders(
        X_train, y_train, X_val, y_val
    )

    model = WorkloadLSTM(
        input_size=len(FEATURES),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_classes=len(CLASS_NAMES),
        dropout=DROPOUT,
    ).to(DEVICE)

    counts = np.bincount(
        y_train,
        minlength=len(CLASS_NAMES),
    )

    class_weights = len(y_train) / (
        len(CLASS_NAMES) * np.maximum(counts, 1)
    )

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(
            class_weights,
            dtype=torch.float32,
        ).to(DEVICE)
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_score = -1.0
    best_epoch = 0
    best_state = None
    patience_count = 0

    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()

        total_loss = 0.0
        total_samples = 0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()

            logits = model(batch_X)

            loss = criterion(logits, batch_y)

            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )

            optimizer.step()

            total_loss += loss.item() * len(batch_X)
            total_samples += len(batch_X)

        train_loss = total_loss / total_samples

        val_probs = predict(model, X_val)
        val_predictions = val_probs.argmax(axis=1)

        val_macro_f1 = f1_score(
            y_val,
            val_predictions,
            average="macro",
            zero_division=0,
        )

        val_loss = 0.0
        val_count = 0

        model.eval()

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(DEVICE)
                batch_y = batch_y.to(DEVICE)

                loss = criterion(
                    model(batch_X),
                    batch_y,
                )

                val_loss += loss.item() * len(batch_X)
                val_count += len(batch_X)

        val_loss /= val_count

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "validation_macro_f1": val_macro_f1,
        })

        logger.info(
            "Epoch %02d/%02d | Train loss %.5f | "
            "Validation loss %.5f | Macro F1 %.5f",
            epoch,
            MAX_EPOCHS,
            train_loss,
            val_loss,
            val_macro_f1,
        )

        if val_macro_f1 > best_score:
            best_score = val_macro_f1
            best_epoch = epoch

            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

            patience_count = 0

        else:
            patience_count += 1

        if patience_count >= PATIENCE:
            logger.info("Early stopping triggered.")
            break

    if best_state is None:
        raise RuntimeError("No valid model checkpoint was produced.")

    model.load_state_dict(best_state)

    return model, pd.DataFrame(history), best_epoch


# =========================================================
# SELECT HIGH-RISK THRESHOLD USING VALIDATION ONLY
# =========================================================

def select_risk_threshold(y_val, val_probs):
    high_id = CLASS_TO_ID["HIGH_DEMAND"]

    actual_high = (y_val == high_id).astype(int)
    high_probs = val_probs[:, high_id]

    candidates = np.unique(
        np.concatenate([
            np.arange(0.01, 0.951, 0.01),
            high_probs,
        ])
    )

    candidates = np.sort(candidates)

    eligible = []

    for threshold in candidates:
        alerts = (high_probs >= threshold).astype(int)

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

    # Maximize validation F1, then precision.
    best = max(
        eligible,
        key=lambda item: (item["f1"], item["precision"]),
    )

    return best, pd.DataFrame(eligible)


# =========================================================
# EVALUATION
# =========================================================

def evaluate_test(y_test, test_probs, threshold):
    predicted = test_probs.argmax(axis=1)

    high_id = CLASS_TO_ID["HIGH_DEMAND"]

    actual_high = (y_test == high_id).astype(int)

    risk_alert = (
        test_probs[:, high_id] >= threshold
    ).astype(int)

    risk_precision = precision_score(
        actual_high,
        risk_alert,
        zero_division=0,
    )

    risk_recall = recall_score(
        actual_high,
        risk_alert,
        zero_division=0,
    )

    risk_f1 = f1_score(
        actual_high,
        risk_alert,
        zero_division=0,
    )

    tn, fp, fn, tp = confusion_matrix(
        actual_high,
        risk_alert,
        labels=[0, 1],
    ).ravel()

    specificity = (
        tn / (tn + fp) if tn + fp else 0.0
    )

    report = {
        "classification_accuracy": float(
            accuracy_score(y_test, predicted)
        ),
        "classification_balanced_accuracy": float(
            balanced_accuracy_score(y_test, predicted)
        ),
        "classification_macro_f1": float(
            f1_score(
                y_test,
                predicted,
                average="macro",
                zero_division=0,
            )
        ),
        "risk_threshold": float(threshold),
        "risk_precision": float(risk_precision),
        "risk_recall": float(risk_recall),
        "risk_f1": float(risk_f1),
        "risk_specificity": float(specificity),
        "risk_true_positives": int(tp),
        "risk_false_positives": int(fp),
        "risk_false_negatives": int(fn),
        "risk_true_negatives": int(tn),
        "risk_alert_count": int(risk_alert.sum()),
    }

    return report, predicted, risk_alert


# =========================================================
# MAIN
# =========================================================

def main():
    set_seed()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Device: %s", DEVICE)

    raw, labels = load_data()

    X, y, metadata = build_sequences(raw, labels)

    (
        X_train, y_train, meta_train,
        X_val, y_val, meta_val,
        X_test, y_test, meta_test,
    ) = chronological_split(X, y, metadata)

    X_train, X_val, X_test, scaler = normalize_sequences(
        X_train, X_val, X_test
    )

    logger.info(
        "Training class distribution: %s",
        dict(zip(CLASS_NAMES, np.bincount(y_train, minlength=3))),
    )

    logger.info(
        "Validation class distribution: %s",
        dict(zip(CLASS_NAMES, np.bincount(y_val, minlength=3))),
    )

    logger.info(
        "Test class distribution: %s",
        dict(zip(CLASS_NAMES, np.bincount(y_test, minlength=3))),
    )

    model, training_history, best_epoch = train_model(
        X_train,
        y_train,
        X_val,
        y_val,
    )

    val_probs = predict(model, X_val)

    best_threshold, threshold_table = select_risk_threshold(
        y_val,
        val_probs,
    )

    logger.info(
        "Selected validation threshold: %s",
        best_threshold,
    )

    test_probs = predict(model, X_test)

    report, predicted, risk_alert = evaluate_test(
        y_test,
        test_probs,
        best_threshold["threshold"],
    )

    test_labels = np.array(CLASS_NAMES)[y_test]
    predicted_labels = np.array(CLASS_NAMES)[predicted]

    predictions = meta_test.copy()

    predictions["predicted_workload"] = predicted_labels
    predictions["high_demand_probability"] = (
        test_probs[:, CLASS_TO_ID["HIGH_DEMAND"]]
    )
    predictions["risk_threshold"] = best_threshold["threshold"]
    predictions["risk_alert"] = np.where(
        risk_alert == 1,
        "HIGH_RISK",
        "NO_HIGH_RISK",
    )

    for index, class_name in enumerate(CLASS_NAMES):
        predictions[f"probability_{class_name}"] = (
            test_probs[:, index]
        )

    predictions["classification_correct"] = (
        predictions["target"] == predictions["predicted_workload"]
    )

    predictions.to_csv(
        OUTPUT_DIR / "test_predictions.csv",
        index=False,
    )

    training_history.to_csv(
        OUTPUT_DIR / "training_history.csv",
        index=False,
    )

    threshold_table.to_csv(
        OUTPUT_DIR / "validation_threshold_analysis.csv",
        index=False,
    )

    pd.DataFrame(
        confusion_matrix(
            y_test,
            predicted,
            labels=[0, 1, 2],
        ),
        index=CLASS_NAMES,
        columns=CLASS_NAMES,
    ).to_csv(
        OUTPUT_DIR / "classification_confusion_matrix.csv"
    )

    pd.DataFrame(
        classification_report(
            y_test,
            predicted,
            labels=[0, 1, 2],
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        )
    ).transpose().to_csv(
        OUTPUT_DIR / "classification_report.csv"
    )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "input_features": FEATURES,
        "class_names": CLASS_NAMES,
        "history_window": HISTORY_WINDOW,
        "forecast_horizon": FORECAST_HORIZON,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "dropout": DROPOUT,
        "best_epoch": best_epoch,
        "risk_threshold": best_threshold["threshold"],
    }

    torch.save(
        checkpoint,
        OUTPUT_DIR / "lstm_workload_classifier.pt",
    )

    np.savez(
        OUTPUT_DIR / "feature_scaler.npz",
        mean=scaler.mean_,
        scale=scaler.scale_,
        var=scaler.var_,
    )

    summary = {
        "model": "LSTM Sequence Classifier",
        "device": str(DEVICE),
        "input_shape": [
            HISTORY_WINDOW,
            len(FEATURES),
        ],
        "training_samples": len(X_train),
        "validation_samples": len(X_val),
        "test_samples": len(X_test),
        "purged_samples": (
            len(X) - len(X_train) - len(X_val) - len(X_test)
        ),
        "best_epoch": best_epoch,
        "validation_selected_threshold": best_threshold,
        "test_metrics": report,
        "classification_report": classification_report(
            y_test,
            predicted,
            labels=[0, 1, 2],
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        ),
    }

    with open(
        OUTPUT_DIR / "lstm_evaluation_report.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=4)

    logger.info(
        "\nLSTM TEST RESULTS\n%s",
        json.dumps(report, indent=4),
    )

    logger.info(
        "\nCLASSIFICATION REPORT\n%s",
        classification_report(
            y_test,
            predicted,
            labels=[0, 1, 2],
            target_names=CLASS_NAMES,
            zero_division=0,
        ),
    )

    logger.info("=" * 65)
    logger.info("LSTM TRAINING AND EVALUATION COMPLETED")
    logger.info("Best epoch: %s", best_epoch)
    logger.info("Output: %s", OUTPUT_DIR)
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
