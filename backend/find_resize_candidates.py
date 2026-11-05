
import csv
import statistics
from pathlib import Path

DATASET = (
    Path(__file__).resolve().parent.parent
    / "datasets"
    / "processed"
    / "server_usage_clean.csv"
)

THRESHOLD = 35.0
MAX_VARIABILITY = 30.0
WINDOW = 12

FEATURES = ["util_CPU", "util_memory"]

def analyze(values):
    current = values[-1]
    average = statistics.mean(values)
    peak = max(values)

    previous_avg = statistics.mean(values[-6:-3])
    recent_avg = statistics.mean(values[-3:])

    change = recent_avg - previous_avg

    if change > 1.0:
        trend = "INCREASING"
    elif change < -1.0:
        trend = "DECREASING"
    else:
        trend = "STABLE"

    std = statistics.pstdev(values)
    cv = (std / average * 100) if average > 0 else None

    return {
        "current": current,
        "average": average,
        "peak": peak,
        "trend": trend,
        "cv": cv,
    }


def main():
    if not DATASET.exists():
        print("Dataset not found:", DATASET)
        return

    machines = {}

    with open(DATASET, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            machine_id = int(row["machineID"])

            observation = {
                "step": int(row["step"]),
                "cpu": float(row["util_CPU"]),
                "memory": float(row["util_memory"]),
            }

            machines.setdefault(machine_id, []).append(observation)

    candidates = []

    for machine_id, observations in machines.items():
        observations.sort(key=lambda x: x["step"])

        if len(observations) < WINDOW:
            continue

        latest = observations[-WINDOW:]

        results = {}

        eligible = True

        for resource in FEATURES:
            key = "cpu" if resource == "util_CPU" else "memory"

            values = [item[key] for item in latest]
            result = analyze(values)

            results[key] = result

            if (
                result["current"] > THRESHOLD
                or result["average"] > THRESHOLD
                or result["peak"] > THRESHOLD
                or result["trend"] == "INCREASING"
                or result["cv"] is None
                or result["cv"] > MAX_VARIABILITY
            ):
                eligible = False

        if eligible:
            candidates.append({
                "machine_id": machine_id,
                "cpu": results["cpu"],
                "memory": results["memory"],
                "observations": len(observations),
            })

    candidates.sort(
        key=lambda x: (
            max(x["cpu"]["peak"], x["memory"]["peak"]),
            max(x["cpu"]["cv"], x["memory"]["cv"]),
        )
    )

    print("\nTotal machines:", len(machines))
    print("RESIZE-eligible candidates:", len(candidates))

    print("\nTop 20 candidates:")
    print("-" * 90)

    for item in candidates[:20]:
        print(f"\nMachine ID: {item['machine_id']}")

        for resource in ("cpu", "memory"):
            r = item[resource]

            print(
                f"{resource.upper():6} | "
                f"Current: {r['current']:.2f}% | "
                f"Average: {r['average']:.2f}% | "
                f"Peak: {r['peak']:.2f}% | "
                f"Trend: {r['trend']} | "
                f"CV: {r['cv']:.2f}%"
            )


if __name__ == "__main__":
    main()
