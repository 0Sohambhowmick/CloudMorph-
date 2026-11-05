
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

ANALYZER_NAME = "CloudMorph Anomaly Detection Engine"
ANALYZER_VERSION = "1.0.0"

WINDOW_SIZE = 12

FEATURE_INDEX = {
    "cpu": 0,
    "memory": 1,
    "disk": 2,
}

FEATURE_NAMES = {
    "cpu": "CPU",
    "memory": "Memory",
    "disk": "Disk",
}

SATURATION_THRESHOLD = 85.0
CRITICAL_SATURATION_THRESHOLD = 95.0

MINIMUM_ABSOLUTE_CHANGE = 5.0
ROBUST_Z_THRESHOLD = 3.0
CRITICAL_ROBUST_Z_THRESHOLD = 5.0

MINIMUM_ROBUST_SCALE = 2.0


# ============================================================
# INPUT VALIDATION
# ============================================================

def validate_history(history):
    """Validate the 12 historical observations."""

    try:
        values = np.asarray(history, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Historical observations must contain numeric values."
        ) from exc

    expected_shape = (WINDOW_SIZE, 6)

    if values.shape != expected_shape:
        raise ValueError(
            f"Expected history shape {expected_shape}; "
            f"received {values.shape}."
        )

    if not np.isfinite(values).all():
        raise ValueError(
            "Historical observations contain invalid values."
        )

    if np.any(values[:, :3] < 0) or np.any(values[:, :3] > 100):
        raise ValueError(
            "CPU, memory and disk utilization must be between 0 and 100."
        )

    if np.any(values[:, 3:] < 0):
        raise ValueError(
            "Load values cannot be negative."
        )

    return values


# ============================================================
# ROBUST STATISTICAL ANALYSIS
# ============================================================

def analyze_resource_anomaly(values, resource):
    """
    Compare the latest observation against the preceding
    11 observations using median and MAD.
    """

    baseline = np.asarray(values[:-1], dtype=np.float64)
    current = float(values[-1])

    baseline_median = float(np.median(baseline))

    absolute_deviations = np.abs(
        baseline - baseline_median
    )

    mad = float(np.median(absolute_deviations))

    robust_scale = max(
        1.4826 * mad,
        MINIMUM_ROBUST_SCALE,
    )

    change = current - baseline_median

    robust_z_score = change / robust_scale

    anomalies = []

    # --------------------------------------------------------
    # SATURATION DETECTION
    # --------------------------------------------------------

    if current >= CRITICAL_SATURATION_THRESHOLD:
        anomalies.append({
            "type": "CRITICAL_RESOURCE_SATURATION",
            "severity": "HIGH",
            "description": (
                f"{FEATURE_NAMES[resource]} utilization is "
                f"{current:.2f}%, reaching critical saturation."
            ),
        })

    elif current >= SATURATION_THRESHOLD:
        anomalies.append({
            "type": "RESOURCE_SATURATION",
            "severity": "MEDIUM",
            "description": (
                f"{FEATURE_NAMES[resource]} utilization is "
                f"{current:.2f}%, exceeding the saturation threshold."
            ),
        })

    # --------------------------------------------------------
    # SUDDEN CHANGE DETECTION
    # --------------------------------------------------------

    if (
        abs(change) >= MINIMUM_ABSOLUTE_CHANGE
        and abs(robust_z_score) >= ROBUST_Z_THRESHOLD
    ):

        direction = (
            "increase" if change > 0 else "decrease"
        )

        severity = (
            "HIGH"
            if abs(robust_z_score) >= CRITICAL_ROBUST_Z_THRESHOLD
            and abs(change) >= 20.0
            else "MEDIUM"
        )

        anomalies.append({
            "type": f"SUDDEN_{direction.upper()}",
            "severity": severity,
            "description": (
                f"{FEATURE_NAMES[resource]} changed by "
                f"{change:+.2f} percentage points compared "
                f"with its historical median."
            ),
        })

    return {
        "resource": resource.upper(),
        "current_utilization": round(current, 2),
        "historical_median": round(baseline_median, 2),
        "change_percentage_points": round(change, 2),
        "robust_z_score": round(robust_z_score, 2),
        "robust_scale": round(robust_scale, 2),
        "is_anomaly": len(anomalies) > 0,
        "anomalies": anomalies,
    }


# ============================================================
# SEVERITY AND RECOMMENDATIONS
# ============================================================

def generate_summary(resource_results):

    detected = [
        item
        for item in resource_results.values()
        if item["is_anomaly"]
    ]

    if not detected:
        return {
            "status": "NORMAL",
            "severity": "LOW",
            "title": "No significant resource anomalies detected",
            "recommendations": [
                "Continue monitoring CPU, memory and disk utilization.",
            ],
        }

    anomaly_types = [
        anomaly["type"]
        for result in detected
        for anomaly in result["anomalies"]
    ]

    severities = [
        anomaly["severity"]
        for result in detected
        for anomaly in result["anomalies"]
    ]

    if "HIGH" in severities:
        severity = "HIGH"
    else:
        severity = "MEDIUM"

    affected = [
        result["resource"]
        for result in detected
    ]

    combined = (
        "CPU" in affected and "MEMORY" in affected
    )

    recommendations = []

    if combined:
        recommendations.append(
            "CPU and memory anomalies occurred together. "
            "Investigate correlated workload changes."
        )

    if "HIGH" in severities:
        recommendations.append(
            "Review resource capacity and application health immediately."
        )

    if any(
        "INCREASE" in anomaly_type
        for anomaly_type in anomaly_types
    ):
        recommendations.append(
            "Inspect recent workload growth and check for resource pressure."
        )

    if any(
        "DECREASE" in anomaly_type
        for anomaly_type in anomaly_types
    ):
        recommendations.append(
            "Investigate sudden workload drops or possible service inactivity."
        )

    if any(
        "SATURATION" in anomaly_type
        for anomaly_type in anomaly_types
    ):
        recommendations.append(
            "Review capacity requirements before considering scaling actions."
        )

    return {
        "status": "ANOMALY_DETECTED",
        "severity": severity,
        "title": "Unusual resource behavior detected",
        "affected_resources": affected,
        "combined_cpu_memory_anomaly": combined,
        "recommendations": recommendations,
    }


# ============================================================
# MAIN ANOMALY DETECTOR
# ============================================================

def detect_anomalies(history):
    """
    Detect anomalies in the latest observation.

    Input:
        12 observations with six features:
        CPU, memory, disk, load1, load5, load15.

    Output:
        Resource-level anomaly analysis and recommendations.

    This engine is advisory only.
    It does not modify infrastructure resources.
    """

    values = validate_history(history)

    resource_results = {}

    for resource, index in FEATURE_INDEX.items():

        resource_results[resource] = analyze_resource_anomaly(
            values[:, index],
            resource,
        )

    summary = generate_summary(resource_results)

    return {
        "status": "success",
        "engine": ANALYZER_NAME,
        "engine_version": ANALYZER_VERSION,
        "analysis_type": "historical_statistical_anomaly_detection",
        "input_metadata": {
            "observations": WINDOW_SIZE,
            "baseline_observations": WINDOW_SIZE - 1,
            "analyzed_observation": "latest",
            "method": "median_absolute_deviation",
        },
        "resource_analysis": resource_results,
        "summary": summary,
        "analysis_metadata": {
            "saturation_threshold": SATURATION_THRESHOLD,
            "critical_saturation_threshold": (
                CRITICAL_SATURATION_THRESHOLD
            ),
            "minimum_absolute_change": MINIMUM_ABSOLUTE_CHANGE,
            "robust_z_threshold": ROBUST_Z_THRESHOLD,
            "automatic_resource_changes": False,
            "requires_manual_validation": True,
        },
    }
