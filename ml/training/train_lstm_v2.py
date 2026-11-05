
import json
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SEQUENCE_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "processed"
    / "sequences"
    / "server_lstm_sequences.npz"
)

MODEL_DIR = PROJECT_ROOT / "ml" / "models"

MODEL_PATH = MODEL_DIR / "server_lstm_v2_best.pt"
METRICS_PATH = MODEL_DIR / "training_metrics_v2.json"
HISTORY_PATH = MODEL_DIR / "training_history_v2.json"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

INPUT_SIZE = 6
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.2

BATCH_SIZE = 512
LEARNING_RATE = 0.0005
MAX_EPOCHS = 30
PATIENCE = 5

TARGET_NAMES = [
    "CPU utilization",
    "Memory utilization",
]


# ============================================================
# REPRODUCIBILITY
# ============================================================

torch.manual_seed(42)
np.random.seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


# ============================================================
# MODEL
# ============================================================

class CloudMorphLSTM(nn.Module):

    def __init__(self):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True,
            dropout=DROPOUT,
        )

        self.dropout = nn.Dropout(DROPOUT)

        self.fc = nn.Linear(
            HIDDEN_SIZE,
            2,
        )

    def forward(self, x):

        output, _ = self.lstm(x)

        last_output = output[:, -1, :]

        last_output = self.dropout(last_output)

        return self.fc(last_output)


# ============================================================
# LOAD DATA
# ============================================================

def load_sequences():

    if not SEQUENCE_PATH.exists():
        raise FileNotFoundError(
            f"Sequence dataset not found: {SEQUENCE_PATH}"
        )

    data = np.load(SEQUENCE_PATH)

    X_train = data["X_train"].astype(np.float32)
    y_train = data["y_train"].astype(np.float32)

    X_val = data["X_val"].astype(np.float32)
    y_val = data["y_val"].astype(np.float32)

    X_test = data["X_test"].astype(np.float32)
    y_test = data["y_test"].astype(np.float32)

    if X_train.shape[1:] != (24, INPUT_SIZE):
        raise ValueError(
            f"Expected input shape (*, 24, 6), got {X_train.shape}"
        )

    if y_train.shape[1] != 2:
        raise ValueError("Expected two target features.")

    return (
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
    )


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_data(
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
):

    # Calculate feature statistics using training data only.
    feature_mean = X_train.reshape(
        -1, INPUT_SIZE
    ).mean(axis=0)

    feature_std = X_train.reshape(
        -1, INPUT_SIZE
    ).std(axis=0)

    feature_std[feature_std == 0] = 1.0

    # Calculate target statistics using training targets only.
    target_mean = y_train.mean(axis=0)
    target_std = y_train.std(axis=0)

    target_std[target_std == 0] = 1.0

    def scale_x(X):
        return (
            (X - feature_mean) / feature_std
        ).astype(np.float32)

    def scale_y(y):
        return (
            (y - target_mean) / target_std
        ).astype(np.float32)

    normalized = {
        "X_train": scale_x(X_train),
        "y_train": scale_y(y_train),

        "X_val": scale_x(X_val),
        "y_val": scale_y(y_val),

        "X_test": scale_x(X_test),
        "y_test": scale_y(y_test),
    }

    scalers = {
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
        "target_mean": target_mean.tolist(),
        "target_std": target_std.tolist(),
    }

    return normalized, scalers


# ============================================================
# TRAINING
# ============================================================

def train_one_epoch(model, loader, optimizer, criterion):

    model.train()

    total_loss = 0.0

    for batch_x, batch_y in loader:

        batch_x = batch_x.to(DEVICE)
        batch_y = batch_y.to(DEVICE)

        optimizer.zero_grad()

        predictions = model(batch_x)

        loss = criterion(
            predictions,
            batch_y,
        )

        loss.backward()

        # Avoid excessively large gradients.
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        total_loss += loss.item() * len(batch_x)

    return total_loss / len(loader.dataset)


# ============================================================
# VALIDATION
# ============================================================

def validate(model, loader, criterion):

    model.eval()

    total_loss = 0.0

    with torch.no_grad():

        for batch_x, batch_y in loader:

            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            predictions = model(batch_x)

            loss = criterion(
                predictions,
                batch_y,
            )

            total_loss += loss.item() * len(batch_x)

    return total_loss / len(loader.dataset)


# ============================================================
# PREDICTION
# ============================================================

def predict(model, X):

    model.eval()

    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32)
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    outputs = []

    with torch.no_grad():

        for (batch_x,) in loader:

            batch_x = batch_x.to(DEVICE)

            predictions = model(batch_x)

            outputs.append(
                predictions.cpu().numpy()
            )

    return np.concatenate(outputs, axis=0)


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(actual, predicted):

    return {
        "MAE": float(
            mean_absolute_error(actual, predicted)
        ),
        "RMSE": float(
            np.sqrt(mean_squared_error(actual, predicted))
        ),
    }


def evaluate_model(model, X_test, y_test_scaled, scalers):

    predicted_scaled = predict(
        model,
        X_test,
    )

    target_mean = np.array(
        scalers["target_mean"]
    )

    target_std = np.array(
        scalers["target_std"]
    )

    # Restore original utilization percentages.
    actual = (
        y_test_scaled * target_std
        + target_mean
    )

    predicted = (
        predicted_scaled * target_std
        + target_mean
    )

    # Persistence baseline: latest observed CPU and memory.
    feature_mean = np.array(
        scalers["feature_mean"]
    )

    feature_std = np.array(
        scalers["feature_std"]
    )

    latest_input = X_test[:, -1, :]

    baseline = (
        latest_input[:, :2] * feature_std[:2]
        + feature_mean[:2]
    )

    clipped_predictions = np.clip(
        predicted,
        0,
        100,
    )

    results = {}

    for index, name in enumerate(TARGET_NAMES):

        results[name] = {
            "LSTM": calculate_metrics(
                actual[:, index],
                predicted[:, index],
            ),

            "LSTM clipped": calculate_metrics(
                actual[:, index],
                clipped_predictions[:, index],
            ),

            "Persistence baseline": calculate_metrics(
                actual[:, index],
                baseline[:, index],
            ),
        }

    return results


# ============================================================
# MAIN
# ============================================================

def main():

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 65)
    print("CLOUDMORPH - LSTM VERSION 2 TRAINING")
    print("=" * 65)

    print("Device:", DEVICE)

    (
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
    ) = load_sequences()

    print("\nLoaded sequences:")

    print("Training:", X_train.shape, y_train.shape)
    print("Validation:", X_val.shape, y_val.shape)
    print("Testing:", X_test.shape, y_test.shape)

    data, scalers = normalize_data(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
    )

    train_dataset = TensorDataset(
        torch.tensor(data["X_train"]),
        torch.tensor(data["y_train"]),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_dataset = TensorDataset(
        torch.tensor(data["X_val"]),
        torch.tensor(data["y_val"]),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = CloudMorphLSTM().to(DEVICE)

    print("\nModel architecture:")
    print(model)

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-5,
    )

    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None

    epochs_without_improvement = 0

    history = []

    print("\nStarting Version 2 training...\n")

    for epoch in range(1, MAX_EPOCHS + 1):

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
        )

        val_loss = validate(
            model,
            val_loader,
            criterion,
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": val_loss,
        })

        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Validation Loss: {val_loss:.6f}"
        )

        if val_loss < best_val_loss:

            best_val_loss = val_loss
            best_epoch = epoch

            best_state = copy.deepcopy(
                model.state_dict()
            )

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:

            print(
                f"\nEarly stopping at epoch {epoch}"
            )

            break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")

    model.load_state_dict(best_state)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "input_size": INPUT_SIZE,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
            "window_size": 24,
            "forecast_horizon": 1,
            "target_names": TARGET_NAMES,
        },
        "scalers": scalers,
        "best_epoch": best_epoch,
        "best_validation_loss": best_val_loss,
    }

    torch.save(
        checkpoint,
        MODEL_PATH,
    )

    print("\nTraining completed.")
    print("Best epoch:", best_epoch)
    print("Best validation loss:", best_val_loss)

    print("\nEvaluating on unseen test data...")

    metrics = evaluate_model(
        model,
        data["X_test"],
        data["y_test"],
        scalers,
    )

    print("\nVERSION 2 TEST RESULTS")

    for target, scores in metrics.items():

        print(f"\n{target}")

        for method, values in scores.items():

            print(
                f"  {method}: "
                f"MAE={values['MAE']:.4f}, "
                f"RMSE={values['RMSE']:.4f}"
            )

    metrics_output = {
        "experiment": "LSTM Version 2",
        "window_size": 24,
        "forecast_horizon": 1,
        "best_epoch": best_epoch,
        "best_validation_loss": best_val_loss,
        "architecture": {
            "input_size": INPUT_SIZE,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
        },
        "test_metrics": metrics,
    }

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metrics_output,
            file,
            indent=4,
        )

    with open(
        HISTORY_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            history,
            file,
            indent=4,
        )

    print("\nVERSION 2 MODEL SAVED:")
    print(MODEL_PATH)

    print("\nVERSION 2 METRICS SAVED:")
    print(METRICS_PATH)

    print("\nVERSION 2 HISTORY SAVED:")
    print(HISTORY_PATH)

    print("\nCLOUDMORPH LSTM VERSION 2 FINISHED")


if __name__ == "__main__":
    main()
