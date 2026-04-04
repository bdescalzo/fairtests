import numpy as np

from metrics.metrics import StandardMetrics, _flatten_probs, _to_numpy


DEFAULT_SWEEP_POINT_COUNT = 101


def _as_binary_labels(y_true):
    return _to_numpy(y_true).astype(int).reshape(-1)


def compute_roc_curve_data(y_true, y_prob):
    y_true = _as_binary_labels(y_true)
    y_prob = _flatten_probs(y_prob)

    if y_true.size == 0:
        return {
            "fpr": [],
            "tpr": [],
            "thresholds": [],
            "auc": float("nan"),
            "n_points": 0,
        }

    order = np.argsort(-y_prob, kind="mergesort")
    y_true_sorted = y_true[order]
    y_prob_sorted = y_prob[order]

    distinct_value_indices = np.where(np.diff(y_prob_sorted))[0]
    threshold_indices = np.r_[distinct_value_indices, y_true_sorted.size - 1]

    tps = np.cumsum(y_true_sorted == 1)[threshold_indices]
    fps = (1 + threshold_indices) - tps

    tps = np.r_[0, tps]
    fps = np.r_[0, fps]
    thresholds = np.r_[np.inf, y_prob_sorted[threshold_indices]]

    n_positive = int(np.sum(y_true == 1))
    n_negative = int(np.sum(y_true == 0))

    if n_positive > 0:
        tpr = tps / n_positive
    else:
        tpr = np.full_like(tps, np.nan, dtype=np.float64)

    if n_negative > 0:
        fpr = fps / n_negative
    else:
        fpr = np.full_like(fps, np.nan, dtype=np.float64)

    if n_positive > 0 and n_negative > 0:
        auc = float(np.trapz(tpr, fpr))
    else:
        auc = float("nan")

    return {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thresholds.tolist(),
        "auc": auc,
        "n_points": int(thresholds.size),
    }


def compute_accuracy_threshold_sweep(
    y_true,
    y_prob,
    sensitive_labels,
    point_count=DEFAULT_SWEEP_POINT_COUNT,
):
    y_true = _as_binary_labels(y_true)
    y_prob = _flatten_probs(y_prob)
    sensitive = _to_numpy(sensitive_labels)
    thresholds = np.linspace(0.0, 1.0, int(point_count), dtype=np.float64)

    standard = StandardMetrics(y_true, y_prob)
    overall_accuracy = []
    worst_group_accuracy = []
    worst_group_id = []

    for threshold in thresholds:
        standard.threshold = float(threshold)
        overall_metrics = standard.compute()
        group_metrics = standard.by_group(sensitive)

        valid_group_accuracies = [
            (group_id, metrics["accuracy"])
            for group_id, metrics in group_metrics.items()
            if metrics.get("accuracy") == metrics.get("accuracy")
        ]

        overall_accuracy.append(float(overall_metrics["accuracy"]))
        if valid_group_accuracies:
            group_id, group_accuracy = min(valid_group_accuracies, key=lambda x: x[1])
            worst_group_accuracy.append(float(group_accuracy))
            worst_group_id.append(group_id)
        else:
            worst_group_accuracy.append(float("nan"))
            worst_group_id.append(None)

    return {
        "thresholds": thresholds.tolist(),
        "accuracy": overall_accuracy,
        "worst_group_accuracy": worst_group_accuracy,
        "worst_group_id": worst_group_id,
        "n_points": int(thresholds.size),
    }


def compute_threshold_curve_bundle(
    y_true,
    y_prob,
    sensitive_labels,
    point_count=DEFAULT_SWEEP_POINT_COUNT,
):
    return {
        "roc": compute_roc_curve_data(y_true, y_prob),
        "accuracy_threshold_sweep": compute_accuracy_threshold_sweep(
            y_true,
            y_prob,
            sensitive_labels,
            point_count=point_count,
        ),
    }
