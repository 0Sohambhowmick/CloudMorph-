import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT / "backend"

# Add both project root and backend to Python import paths
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BACKEND_DIR))

from app.services import resource_optimizer as optimizer


def make_workload(high_pressure=False):
    cpu = 90 if high_pressure else 20
    memory = 90 if high_pressure else 20

    return {
        "status": "success",
        "resource_statistics": {
            "cpu": {
                "current": cpu,
                "average": cpu,
                "peak": cpu,
            },
            "memory": {
                "current": memory,
                "average": memory,
                "peak": memory,
            },
        },
        "workload_trends": {
            "cpu": {"direction": "STABLE"},
            "memory": {"direction": "STABLE"},
        },
        "workload_variability": {
            "cpu": {"coefficient_of_variation": 5},
            "memory": {"coefficient_of_variation": 5},
        },
        "forecasts": {
            "15min": {
                "cpu_utilization": cpu,
                "memory_utilization": memory,
            },
            "30min": {
                "cpu_utilization": cpu,
                "memory_utilization": memory,
            },
        },
    }


history = [[20, 20, 40, 1, 1, 1] for _ in range(12)]


def set_prediction(condition):
    optimizer.predict_workload_condition = lambda records: {
        "predicted_condition": condition,
        "confidence": 0.90,
    }


high_risk = {
    "status": "success",
    "high_memory_risk": True,
    "risk_probability": 0.80,
    "risk_probability_percent": 80,
}

no_risk = {
    "status": "success",
    "high_memory_risk": False,
    "risk_probability": 0.0,
    "risk_probability_percent": 0.0,
}

unavailable_risk = {
    "status": "unavailable",
}


# TEST 1: High resource pressure must preserve SCALE_UP.
set_prediction("NORMAL_DEMAND")

result = optimizer.optimize_resources(
    make_workload(high_pressure=True),
    history,
    high_risk,
)

assert result["recommendation"]["action"] == "SCALE_UP"
print("TEST 1 PASSED: High pressure preserves SCALE_UP")


# TEST 2: High memory risk must block resource reduction.
set_prediction("LOW_DEMAND")

result = optimizer.optimize_resources(
    make_workload(),
    history,
    high_risk,
)

assert result["recommendation"]["action"] == "KEEP"
assert result["recommendation"]["priority"] == "HIGH"
assert result["decision_source"] == "ml_memory_high_risk_monitoring"
assert result["analysis_metadata"]["memory_risk_model_used"] is True

print("TEST 2 PASSED: High memory risk produces KEEP / HIGH")


# TEST 3: Low demand and no memory risk should allow safe RESIZE.
result = optimizer.optimize_resources(
    make_workload(),
    history,
    no_risk,
)

assert result["recommendation"]["action"] == "RESIZE"
print("TEST 3 PASSED: Safe low demand permits RESIZE")


# TEST 4: Unavailable memory-risk model must not break optimization.
result = optimizer.optimize_resources(
    make_workload(),
    history,
    unavailable_risk,
)

assert result["recommendation"]["action"] == "RESIZE"
assert result["analysis_metadata"]["memory_risk_model_available"] is False

print("TEST 4 PASSED: Unavailable memory model does not block optimization")

print("\nALL FOUR OPTIMIZER INTEGRATION TESTS PASSED.")
