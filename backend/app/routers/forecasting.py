
import sys
import csv
from pathlib import Path
from functools import lru_cache

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATASET_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "processed"
    / "server_usage_clean.csv"
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.inference import forecast_utilization

# --------------------------------------------------
# ROUTER
# --------------------------------------------------

router = APIRouter(
    prefix="/forecast",
    tags=["Forecasting"]
)

# --------------------------------------------------
# FEATURE CONFIGURATION
# --------------------------------------------------

FEATURE_NAMES = [
    "util_CPU",
    "util_memory",
    "util_disk",
    "load1",
    "load5",
    "load15",
]

WINDOW_SIZE = 12


# --------------------------------------------------
# REQUEST MODEL
# --------------------------------------------------

class ForecastRequest(BaseModel):
    history: list[list[float]] = Field(
        ...,
        description=(
            "Exactly 12 historical observations, "
            "each containing six resource features."
        )
    )


# --------------------------------------------------
# LOAD AND CACHE HISTORICAL DATASET
# --------------------------------------------------

@lru_cache(maxsize=1)
def load_historical_dataset():
    """
    Load the cleaned Alibaba server usage dataset.
    The dataset is cached to avoid reading the CSV
    repeatedly for every API request.
    """

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Historical dataset not found: {DATASET_PATH}"
        )

    machine_data = {}

    with open(
        DATASET_PATH,
        mode="r",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        reader = csv.DictReader(csv_file)

        required_columns = {
            "machineID",
            "timestamp",
            "step",
            *FEATURE_NAMES,
        }

        available_columns = set(reader.fieldnames or [])

        missing_columns = required_columns - available_columns

        if missing_columns:
            raise ValueError(
                f"Dataset is missing columns: {sorted(missing_columns)}"
            )

        for row in reader:

            machine_id = int(row["machineID"])

            observation = {
                "timestamp": int(row["timestamp"]),
                "step": int(row["step"]),
                "features": [
                    float(row[feature])
                    for feature in FEATURE_NAMES
                ],
            }

            machine_data.setdefault(
                machine_id, []
            ).append(observation)

    # Sort observations chronologically for each machine.
    for machine_id in machine_data:
        machine_data[machine_id].sort(
            key=lambda item: item["step"]
        )

    return machine_data


# --------------------------------------------------
# GET HISTORICAL OBSERVATIONS
# --------------------------------------------------

@router.get("/history")
async def get_historical_data(
    machine_id: int = Query(default=1, ge=1)
):
    """
    Return the latest 12 historical observations
    for the requested machine.
    """

    try:
        dataset = load_historical_dataset()

        if machine_id not in dataset:
            raise HTTPException(
                status_code=404,
                detail=f"Machine {machine_id} was not found."
            )

        observations = dataset[machine_id]

        if len(observations) < WINDOW_SIZE:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Machine {machine_id} does not have "
                    f"{WINDOW_SIZE} observations."
                )
            )

        latest_observations = observations[-WINDOW_SIZE:]

        history = [
            observation["features"]
            for observation in latest_observations
        ]

        return {
            "status": "success",
            "source": "Alibaba Cluster Trace 2017",
            "machine_id": machine_id,
            "observations": len(history),
            "feature_names": FEATURE_NAMES,
            "history": history,
            "start_timestamp": latest_observations[0]["timestamp"],
            "end_timestamp": latest_observations[-1]["timestamp"],
        }

    except HTTPException:
        raise

    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail="Historical forecasting dataset is unavailable."
        )

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Unable to load historical resource observations."
        )


# --------------------------------------------------
# GENERATE FORECAST
# --------------------------------------------------

@router.post("/predict")
async def predict_forecast(request: ForecastRequest):

    try:
        history_array = np.asarray(
            request.history,
            dtype=np.float32
        )

        result = forecast_utilization(history_array)

        return result

    except (ValueError, TypeError) as error:
        raise HTTPException(
            status_code=422,
            detail=str(error)
        )

    except Exception:
        raise HTTPException(
            status_code=500,
            detail=(
                "Forecasting failed due to an internal server error."
            )
        )


# --------------------------------------------------
# FORECASTING SERVICE HEALTH
# --------------------------------------------------

@router.get("/health")
async def forecasting_health():

    return {
        "status": "success",
        "module": "CloudMorph Forecasting",
        "forecast_method": "Persistence",
        "model_status": "baseline",
        "supported_horizons": [
            "15min",
            "30min"
        ],
    }
