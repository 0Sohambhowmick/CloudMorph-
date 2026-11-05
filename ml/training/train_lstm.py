
from pathlib import Path
import json
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

SEQUENCE_FILE = (
    ROOT / "datasets" / "processed"
    / "sequences" / "server_lstm_sequences.npz"
)

MODEL_DIR = ROOT / "ml" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = MODEL_DIR / "server_lstm_best.pt"
METRICS_FILE = MODEL_DIR / "training_metrics.json"
HISTORY_FILE = MODEL_DIR / "training_history.json"

SEED = 42
BATCH_SIZE = 512
EPOCHS = 20
LEARNING_RATE = 0.001
HIDDEN_SIZE = 64
PATIENCE = 4

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# MODEL ARCHITECTURE
# ============================================================

class CloudMorphLSTM(nn.Module):

    def __init__(
        self,
        input_size=6,
        hidden_size=64,
        output_size=2,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )

        self.dropout = nn.Dropout(0.2)

        self.fc = nn.Linear(
            hidden_size,
            output_size,
        )

    def forward(self, x):
        output, _ = self.lstm(x)

        # Use the final time-step representation.
        last_output = output[:, -1, :]

        last_output = self.dropout(last_output)

        return self.fc(last_output)


# ============================================================
# LOAD DATA
# ============================================================

def load_sequences():

    if not SEQUENCE_FILE.exists():
        raise FileNotFoundError(SEQUENCE_FILE)

    data = np.load(SEQUENCE_FILE)

    X_train = data["X_train"].astype(np.float32)
    y_train = data["y_train"].astype(np.float32)

    X_val = data["X_val"].astype(np.float32)
    y_val = data["y_val"].astype(np.float32)

    X_test = data["X_test"].astype(np.float32)
    y_test = data["y_test"].astype(np.float32)

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

    # Fit input scaler ONLY on training observations.
    feature_mean = X_train.reshape(
        -1, X_train.shape[-1]
    ).mean(axis=0)

    feature_std = X_train.reshape(
        -1, X_train.shape[-1]
    ).std(axis=0)

    feature_std = np.where(
        feature_std < 1e-8,
        1.0,
        feature_std,
    )

    # Fit target scaler ONLY on training targets.
    target_mean = y_train.mean(axis=0)
    target_std = y_train.std(axis=0)

    target_std = np.where(
        target_std < 1e-8,
        1.0,
        target_std,
    )

    def scale_x(values):
        return (
            (values - feature_mean) / feature_std
        ).astype(np.float32)

    def scale_y(values):
        return (
            (values - target_mean) / target_std
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

def train_model(model, train_loader, val_loader):

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    history = []

    for epoch in range(EPOCHS):

        # Training phase
        model.train()

        train_loss_sum = 0.0
        train_samples = 0

        for batch_x, batch_y in train_loader:

            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()

            predictions = model(batch_x)

            loss = criterion(
                predictions,
                batch_y,
            )

            loss.backward()

            optimizer.step()

            train_loss_sum += (
                loss.item() * len(batch_x)
            )

            train_samples += len(batch_x)

        train_loss = (
            train_loss_sum / train_samples
        )

        # Validation phase
        model.eval()

        val_loss_sum = 0.0
        val_samples = 0

        with torch.no_grad():

            for batch_x, batch_y in val_loader:

                batch_x = batch_x.to(DEVICE)
                batch_y = batch_y.to(DEVICE)

                predictions = model(batch_x)

                loss = criterion(
                    predictions,
                    batch_y,
                )

                val_loss_sum += (
                    loss.item() * len(batch_x)
                )

                val_samples += len(batch_x)

        val_loss = val_loss_sum / val_samples

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "validation_loss": val_loss,
        })

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Validation Loss: {val_loss:.6f}"
        )

        # Save best model.
        if val_loss < best_val_loss:

            best_val_loss = val_loss

            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

            epochs_without_improvement = 0

        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:

            print(
                f"\nEarly stopping at epoch {epoch + 1}"
            )

            break

    if best_state is None:
        raise RuntimeError("Model training did not produce a valid checkpoint.")

    model.load_state_dict(best_state)

    return model, history, best_val_loss


# ============================================================
# PREDICTIONS AND EVALUATION
# ============================================================

def predict(model, X):

    model.eval()

    loader = DataLoader(
        TensorDataset(
            torch.tensor(X, dtype=torch.float32)
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    predictions = []

    with torch.no_grad():

        for (batch_x,) in loader:

            batch_x = batch_x.to(DEVICE)

            output = model(batch_x)

            predictions.append(
                output.cpu().numpy()
            )

    return np.concatenate(predictions, axis=0)


def calculate_metrics(actual, predicted):

    return {
        "MAE": float(
            mean_absolute_error(actual, predicted)
        ),
        "RMSE": float(
            np.sqrt(mean_squared_error(actual, predicted))
        ),
    }


def evaluate_model(
    model,
    X_test,
    y_test,
    scalers,
):

    # LSTM predictions in normalized scale.
    predicted_scaled = predict(model, X_test)

    target_mean = np.array(
        scalers["target_mean"]
    )

    target_std = np.array(
        scalers["target_std"]
    )

    # Return predictions to actual utilization percentages.
    predicted = (
        predicted_scaled * target_std
        + target_mean
    )

    actual = (
        y_test * target_std
        + target_mean
    )

    # Persistence baseline:
    # Predict next CPU/memory as the latest observed values.
    baseline = X_test[:, -1, :2].copy()

    # Input features were normalized, so inverse-transform
    # using the first two feature statistics.
    feature_mean = np.array(
        scalers["feature_mean"]
    )[:2]

    feature_std = np.array(
        scalers["feature_std"]
    )[:2]

    baseline = (
        baseline * feature_std
        + feature_mean
    )

    target_names = [
        "CPU utilization",
        "Memory utilization",
    ]

    results = {}

    for index, name in enumerate(target_names):

        results[name] = {
            "LSTM": calculate_metrics(
                actual[:, index],
                predicted[:, index],
            ),
            "Persistence baseline": calculate_metrics(
                actual[:, index],
                baseline[:, index],
            ),
        }

    # Keep predictions within the valid utilization range.
    clipped_predictions = np.clip(
        predicted,
        0,
        100,
    )

    results["LSTM_clipped"] = {}

    for index, name in enumerate(target_names):

        results["LSTM_clipped"][name] = (
            calculate_metrics(
                actual[:, index],
                clipped_predictions[:, index],
            )
        )

    return results


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("CLOUDMORPH - LSTM TRAINING")
    print("=" * 65)

    print(f"Device: {DEVICE}")

    (
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
    ) = load_sequences()

    print("\nLoaded sequences:")

    print("Training:", X_train.shape, y_train.shape)
    print("Validation:", X_val.shape, y_val.shape)
    print("Testing:", X_test.shape, y_test.shape)

    # Normalize using training statistics only.
    data, scalers = normalize_data(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
    )

    train_dataset = TensorDataset(
        torch.tensor(data["X_train"]),
        torch.tensor(data["y_train"]),
    )

    val_dataset = TensorDataset(
        torch.tensor(data["X_val"]),
        torch.tensor(data["y_val"]),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = CloudMorphLSTM(
        input_size=6,
        hidden_size=HIDDEN_SIZE,
        output_size=2,
    ).to(DEVICE)

    print("\nModel architecture:")
    print(model)

    print("\nStarting training...\n")

    model, history, best_val_loss = train_model(
        model,
        train_loader,
        val_loader,
    )

    print("\nTraining completed.")
    print(f"Best validation loss: {best_val_loss:.6f}")

    print("\nEvaluating on unseen test data...")

    metrics = evaluate_model(
    model,
    data["X_test"],
    data["y_test"],
    scalers,
    )

    print("\nTEST RESULTS")

    for target, values in metrics.items():

        print(f"\n{target}")

        if target == "LSTM_clipped":
            for name, score in values.items():
                print(
                    f"  {name}: "
                    f"MAE={score['MAE']:.4f}, "
                    f"RMSE={score['RMSE']:.4f}"
                )
        else:
            for method, score in values.items():
                print(
                    f"  {method}: "
                    f"MAE={score['MAE']:.4f}, "
                    f"RMSE={score['RMSE']:.4f}"
                )

    # Save best model and its normalization parameters.
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "input_size": 6,
        "hidden_size": HIDDEN_SIZE,
        "output_size": 2,
        "window_size": 12,
        "forecast_horizon_steps": 1,
        "feature_names": [
            "util_CPU",
            "util_memory",
            "util_disk",
            "load1",
            "load5",
            "load15",
        ],
        "target_names": [
            "util_CPU",
            "util_memory",
        ],
        "scalers": scalers,
        "best_validation_loss": best_val_loss,
        "seed": SEED,
    }

    torch.save(checkpoint, MODEL_FILE)

    with open(
        METRICS_FILE, "w", encoding="utf-8"
    ) as file:
        json.dump(metrics, file, indent=4)

    with open(
        HISTORY_FILE, "w", encoding="utf-8"
    ) as file:
        json.dump(history, file, indent=4)

    print("\nMODEL SAVED:")
    print(MODEL_FILE)

    print("\nMETRICS SAVED:")
    print(METRICS_FILE)

    print("\nTRAINING HISTORY SAVED:")
    print(HISTORY_FILE)

    print("\nCLOUDMORPH LSTM TRAINING FINISHED")


if __name__ == "__main__":
    main()
