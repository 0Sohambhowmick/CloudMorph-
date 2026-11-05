
import copy
import json
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

DATA_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "processed"
    / "sequences"
    / "server_multihorizon_sequences.npz"
)

MODEL_DIR = PROJECT_ROOT / "ml" / "models"

MODEL_PATH = MODEL_DIR / "server_lstm_v4_best.pt"
METRICS_PATH = MODEL_DIR / "training_metrics_v4.json"
HISTORY_PATH = MODEL_DIR / "training_history_v4.json"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

INPUT_SIZE = 6
HIDDEN_SIZE = 32
NUM_LAYERS = 1
DROPOUT = 0.3

BATCH_SIZE = 512
LEARNING_RATE = 0.0003
WEIGHT_DECAY = 1e-4

MAX_EPOCHS = 30
PATIENCE = 5

TARGET_NAMES = [
    "CPU utilization",
    "Memory utilization",
]

HORIZONS = ["15min", "30min"]


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

class CloudMorphMultiHorizonLSTM(nn.Module):

    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True,
        )

        self.dropout = nn.Dropout(DROPOUT)

        # Separate prediction head for each horizon.
        self.head_15 = nn.Linear(HIDDEN_SIZE, 2)
        self.head_30 = nn.Linear(HIDDEN_SIZE, 2)

    def forward(self, x):

        output, _ = self.lstm(x)

        representation = output[:, -1, :]

        representation = self.dropout(representation)

        prediction_15 = self.head_15(representation)
        prediction_30 = self.head_30(representation)

        return prediction_15, prediction_30


# ============================================================
# LOAD DATA
# ============================================================

def load_sequences():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Sequence dataset not found: {DATA_PATH}"
        )

    with np.load(DATA_PATH) as data:

        X_train = data["X_train"].astype(np.float32)
        X_val = data["X_val"].astype(np.float32)
        X_test = data["X_test"].astype(np.float32)

        horizon_suffixes = {
            "15min": "15",
            "30min": "30",
        }

        targets = {}

        for horizon, suffix in horizon_suffixes.items():

            targets[f"y_train_{horizon}"] = (
                data[f"y_train_{suffix}"].astype(np.float32)
            )

            targets[f"y_val_{horizon}"] = (
                data[f"y_val_{suffix}"].astype(np.float32)
            )

            targets[f"y_test_{horizon}"] = (
                data[f"y_test_{suffix}"].astype(np.float32)
            )

    # Validate input shapes and values.
    for split_name, X in [
        ("train", X_train),
        ("validation", X_val),
        ("test", X_test),
    ]:

        if X.ndim != 3 or X.shape[1:] != (12, 6):
            raise ValueError(
                f"Invalid {split_name} input shape: {X.shape}. "
                f"Expected (*, 12, 6)."
            )

        if not np.isfinite(X).all():
            raise ValueError(
                f"NaN or infinite values found in "
                f"{split_name} inputs."
            )

    # Validate target shapes and values.
    for horizon in HORIZONS:

        for split in ["train", "val", "test"]:

            target = targets[f"y_{split}_{horizon}"]

            if target.ndim != 2 or target.shape[1] != 2:
                raise ValueError(
                    f"Invalid target shape for {horizon}, "
                    f"{split}: {target.shape}"
                )

            if not np.isfinite(target).all():
                raise ValueError(
                    f"NaN or infinite values found in "
                    f"y_{split}_{horizon}."
                )

        if len(X_train) != len(targets[f"y_train_{horizon}"]):
            raise ValueError("Training sample count mismatch.")

        if len(X_val) != len(targets[f"y_val_{horizon}"]):
            raise ValueError("Validation sample count mismatch.")

        if len(X_test) != len(targets[f"y_test_{horizon}"]):
            raise ValueError("Testing sample count mismatch.")

    return X_train, X_val, X_test, targets


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_data(X_train, X_val, X_test, targets):

    feature_mean = X_train.reshape(
        -1, INPUT_SIZE
    ).mean(axis=0)

    feature_std = X_train.reshape(
        -1, INPUT_SIZE
    ).std(axis=0)

    feature_std[feature_std == 0] = 1.0

    def scale_x(X):

        return (
            (X - feature_mean) / feature_std
        ).astype(np.float32)

    normalized_targets = {}
    target_scalers = {}

    for horizon in HORIZONS:

        train_y = targets[f"y_train_{horizon}"]

        target_mean = train_y.mean(axis=0)
        target_std = train_y.std(axis=0)

        target_std[target_std == 0] = 1.0

        target_scalers[horizon] = {
            "mean": target_mean.tolist(),
            "std": target_std.tolist(),
        }

        for split in ["train", "val", "test"]:

            key = f"y_{split}_{horizon}"

            normalized_targets[key] = (
                (targets[key] - target_mean) / target_std
            ).astype(np.float32)

    normalized = {
        "X_train": scale_x(X_train),
        "X_val": scale_x(X_val),
        "X_test": scale_x(X_test),
        **normalized_targets,
    }

    scalers = {
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
        "targets": target_scalers,
    }

    return normalized, scalers


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(model, loader, optimizer, criterion):

    model.train()

    total_loss = 0.0

    for batch_x, batch_y15, batch_y30 in loader:

        batch_x = batch_x.to(DEVICE)
        batch_y15 = batch_y15.to(DEVICE)
        batch_y30 = batch_y30.to(DEVICE)

        optimizer.zero_grad()

        prediction_15, prediction_30 = model(batch_x)

        loss_15 = criterion(
            prediction_15,
            batch_y15,
        )

        loss_30 = criterion(
            prediction_30,
            batch_y30,
        )

        # Equal importance for both forecast horizons.
        loss = (loss_15 + loss_30) / 2

        loss.backward()

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

        for batch_x, batch_y15, batch_y30 in loader:

            batch_x = batch_x.to(DEVICE)
            batch_y15 = batch_y15.to(DEVICE)
            batch_y30 = batch_y30.to(DEVICE)

            prediction_15, prediction_30 = model(batch_x)

            loss_15 = criterion(
                prediction_15,
                batch_y15,
            )

            loss_30 = criterion(
                prediction_30,
                batch_y30,
            )

            loss = (loss_15 + loss_30) / 2

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

    predictions_15 = []
    predictions_30 = []

    with torch.no_grad():

        for (batch_x,) in loader:

            batch_x = batch_x.to(DEVICE)

            output_15, output_30 = model(batch_x)

            predictions_15.append(
                output_15.cpu().numpy()
            )

            predictions_30.append(
                output_30.cpu().numpy()
            )

    return (
        np.concatenate(predictions_15, axis=0),
        np.concatenate(predictions_30, axis=0),
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(actual, predicted):

    return {
        "MAE": float(
            mean_absolute_error(actual, predicted)
        ),

        "RMSE": float(
            np.sqrt(
                mean_squared_error(actual, predicted)
            )
        ),
    }


def evaluate_model(model, X_test, y_test, scalers):

    predicted_15_scaled, predicted_30_scaled = predict(
        model,
        X_test,
    )

    predictions = {
        "15min": predicted_15_scaled,
        "30min": predicted_30_scaled,
    }

    feature_mean = np.array(
        scalers["feature_mean"]
    )

    feature_std = np.array(
        scalers["feature_std"]
    )

    # Latest observation available at each forecast origin.
    latest_input = X_test[:, -1, :]

    baseline = (
        latest_input[:, :2] * feature_std[:2]
        + feature_mean[:2]
    )

    results = {}

    for horizon in HORIZONS:

        target_mean = np.array(
            scalers["targets"][horizon]["mean"]
        )

        target_std = np.array(
            scalers["targets"][horizon]["std"]
        )

        actual = (
            y_test[f"y_test_{horizon}"] * target_std
            + target_mean
        )

        predicted = (
            predictions[horizon] * target_std
            + target_mean
        )

        clipped = np.clip(
            predicted,
            0,
            100,
        )

        results[horizon] = {}

        for index, name in enumerate(TARGET_NAMES):

            results[horizon][name] = {

                "LSTM": calculate_metrics(
                    actual[:, index],
                    predicted[:, index],
                ),

                "LSTM clipped": calculate_metrics(
                    actual[:, index],
                    clipped[:, index],
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
    print("CLOUDMORPH - MULTI-HORIZON LSTM V4")
    print("=" * 65)

    print("Device:", DEVICE)

    print("\nV4 Configuration:")
    print("Hidden size:", HIDDEN_SIZE)
    print("Dropout:", DROPOUT)
    print("Learning rate:", LEARNING_RATE)
    print("Weight decay:", WEIGHT_DECAY)

    X_train, X_val, X_test, targets = load_sequences()

    print("\nLoaded sequences:")

    print("Training:", X_train.shape)
    print("Validation:", X_val.shape)
    print("Testing:", X_test.shape)

    normalized, scalers = normalize_data(
        X_train,
        X_val,
        X_test,
        targets,
    )

    train_dataset = TensorDataset(
        torch.tensor(normalized["X_train"]),
        torch.tensor(normalized["y_train_15min"]),
        torch.tensor(normalized["y_train_30min"]),
    )

    val_dataset = TensorDataset(
        torch.tensor(normalized["X_val"]),
        torch.tensor(normalized["y_val_15min"]),
        torch.tensor(normalized["y_val_30min"]),
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

    model = CloudMorphMultiHorizonLSTM().to(DEVICE)

    print("\nModel architecture:")
    print(model)

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )

    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None

    epochs_without_improvement = 0

    history = []

    print("\nStarting training...\n")

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

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "learning_rate": current_lr,
        })

        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Validation Loss: {val_loss:.6f} | "
            f"LR: {current_lr:.7f}"
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
        raise RuntimeError(
            "Training failed to produce a valid model."
        )

    model.load_state_dict(best_state)

    checkpoint = {
        "model_state_dict": model.state_dict(),

        "model_config": {
            "input_size": INPUT_SIZE,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
            "window_size": 12,
            "forecast_horizons": {
                "15min": 3,
                "30min": 6,
            },
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
        normalized["X_test"],
        normalized,
        scalers,
    )

    for horizon, horizon_results in metrics.items():

        print(f"\n{horizon.upper()} FORECAST RESULTS")

        for target, scores in horizon_results.items():

            print(f"\n{target}")

            for method, values in scores.items():

                print(
                    f"  {method}: "
                    f"MAE={values['MAE']:.4f}, "
                    f"RMSE={values['RMSE']:.4f}"
                )

    metrics_output = {
        "experiment": "CloudMorph Multi-Horizon LSTM V4",

        "window_size": 12,
        "forecast_horizons": [15, 30],

        "best_epoch": best_epoch,
        "best_validation_loss": best_val_loss,

        "architecture": {
            "input_size": INPUT_SIZE,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
        },

        "training_configuration": {
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "maximum_epochs": MAX_EPOCHS,
            "early_stopping_patience": PATIENCE,
        },

        "test_sample_count": len(X_test),
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

    print("\nMODEL SAVED:")
    print(MODEL_PATH)

    print("\nMETRICS SAVED:")
    print(METRICS_PATH)

    print("\nTRAINING HISTORY SAVED:")
    print(HISTORY_PATH)

    print("\nCLOUDMORPH MULTI-HORIZON V4 COMPLETED")


if __name__ == "__main__":
    main()
