
import numpy as np

from ml.inference.optimization_classifier import (
    predict_workload_condition,
)


def main():

    # Synthetic example history for inference smoke testing.
    # These values are test inputs, not live cloud telemetry.

    history = []

    for i in range(12):

        history.append({
            "util_CPU": 35 + i * 2,
            "util_memory": 45 + i * 1.5,
            "util_disk": 40,
            "load1": 1.5 + i * 0.1,
            "load5": 1.3 + i * 0.1,
            "load15": 1.1 + i * 0.1,
        })

    result = predict_workload_condition(history)

    print("\n========== CLOUDMORPH ML INFERENCE TEST ==========")

    print("Predicted condition:", result["predicted_condition"])

    print("Confidence:", round(result["confidence"], 4))

    print("Class probabilities:")

    for label, probability in result["class_probabilities"].items():

        print(
            f"  {label}: {probability:.4f}"
        )

    print("Model:", result["model"])

    print("Model version:", result["model_version"])

    print("Prediction horizon:", result["prediction_horizon_minutes"], "minutes")

    print("\nInference completed successfully.")


if __name__ == "__main__":
    main()
