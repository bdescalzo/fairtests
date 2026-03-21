import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch optional for metrics
    torch = None


def _to_numpy(array_like):
    if torch is not None and isinstance(array_like, torch.Tensor):
        return array_like.detach().cpu().numpy()
    return np.asarray(array_like)


def _flatten_probs(y_prob):
    y_prob = _to_numpy(y_prob)
    if y_prob.ndim == 2 and y_prob.shape[1] == 1:
        return y_prob[:, 0]
    return y_prob.reshape(-1)


def _confusion(y_true, y_prob, threshold=0.5):
    y_true = _to_numpy(y_true).astype(int).reshape(-1)
    y_prob = _flatten_probs(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    return tp, fp, tn, fn


def _safe_div(num, den, default=0.0):
    return num / den if den > 0 else default


def _is_nan(value):
    return value != value


def _worst_pairwise_gap(group_metrics, metric_name):
    valid = []
    for group_id, metrics in group_metrics.items():
        value = metrics.get(metric_name, float("nan"))
        if _is_nan(value):
            continue
        valid.append((group_id, value))

    if len(valid) < 2:
        return float("nan"), None

    worst_gap = -1.0
    worst_pair = None
    for idx, (left_group, left_value) in enumerate(valid[:-1]):
        for right_group, right_value in valid[idx + 1 :]:
            gap = abs(left_value - right_value)
            if gap > worst_gap:
                worst_gap = gap
                worst_pair = (left_group, right_group)

    return worst_gap, worst_pair


def _worst_pairwise_ratio(group_metrics, metric_name):
    valid = []
    for group_id, metrics in group_metrics.items():
        value = metrics.get(metric_name, float("nan"))
        if _is_nan(value):
            continue
        valid.append((group_id, value))

    if len(valid) < 2:
        return float("nan"), None

    worst_ratio = float("inf")
    worst_pair = None
    for idx, (left_group, left_value) in enumerate(valid[:-1]):
        for right_group, right_value in valid[idx + 1 :]:
            high = max(left_value, right_value)
            low = min(left_value, right_value)
            ratio = _safe_div(low, high, default=float("nan"))
            if _is_nan(ratio):
                continue
            if ratio < worst_ratio:
                worst_ratio = ratio
                worst_pair = (left_group, right_group)

    if worst_pair is None:
        return float("nan"), None
    return worst_ratio, worst_pair


class StandardMetrics:
    def __init__(self, y_true, y_prob, threshold=0.5):
        self.y_true = y_true
        self.y_prob = y_prob
        self.threshold = threshold

    def compute(self):
        tp, fp, tn, fn = _confusion(self.y_true, self.y_prob, self.threshold)
        n = tp + fp + tn + fn
        accuracy = _safe_div(tp + tn, n, default=float("nan"))
        tpr = _safe_div(tp, tp + fn)
        fpr = _safe_div(fp, fp + tn)
        tnr = _safe_div(tn, tn + fp)
        fnr = _safe_div(fn, fn + tp)
        precision = _safe_div(tp, tp + fp)
        f1 = _safe_div(2 * precision * tpr, precision + tpr)
        balanced_acc = _safe_div(tpr + tnr, 2)
        selection_rate = _safe_div(tp + fp, n, default=float("nan"))

        return {
            "n_members": n,
            "TP": tp,
            "FP": fp,
            "TN": tn,
            "FN": fn,
            "accuracy": accuracy,
            "TPR": tpr,
            "FPR": fpr,
            "TNR": tnr,
            "FNR": fnr,
            "precision": precision,
            "f1": f1,
            "balanced_accuracy": balanced_acc,
            "selection_rate": selection_rate,
        }

    def by_group(self, sensitive_labels):
        sensitive = _to_numpy(sensitive_labels)
        y_true = _to_numpy(self.y_true)
        y_prob = _flatten_probs(self.y_prob)

        results = {}
        for group in np.unique(sensitive):
            mask = sensitive == group
            if mask.sum() == 0:
                continue
            group_metrics = StandardMetrics(
                y_true[mask], y_prob[mask], threshold=self.threshold
            ).compute()
            results[group] = group_metrics
        return results


class FairnessMetrics:
    def __init__(self, y_true, y_prob, sensitive_labels, protected_value, threshold=0.5):
        self.y_true = y_true
        self.y_prob = y_prob
        self.sensitive = _to_numpy(sensitive_labels)
        self.protected_value = protected_value
        self.threshold = threshold

    def compute(self):
        y_true = _to_numpy(self.y_true)
        y_prob = _flatten_probs(self.y_prob)

        group_metrics = StandardMetrics(
            y_true, y_prob, self.threshold
        ).by_group(self.sensitive)
        worst_group_id = None
        worst_group_accuracy = float("nan")
        if group_metrics:
            valid = [
                (gid, metrics["accuracy"])
                for gid, metrics in group_metrics.items()
                if metrics.get("accuracy") == metrics.get("accuracy")
            ]
            if valid:
                worst_group_id, worst_group_accuracy = min(valid, key=lambda x: x[1])

        dp_diff, dp_pair = _worst_pairwise_gap(group_metrics, "selection_rate")
        dp_ratio, dp_ratio_pair = _worst_pairwise_ratio(group_metrics, "selection_rate")
        tpr_diff, tpr_pair = _worst_pairwise_gap(group_metrics, "TPR")
        fpr_diff, fpr_pair = _worst_pairwise_gap(group_metrics, "FPR")
        acc_diff, acc_pair = _worst_pairwise_gap(group_metrics, "accuracy")

        eq_odds = float("nan")
        eq_odds_pair = None
        avg_odds = float("nan")
        avg_odds_pair = None
        group_ids = list(group_metrics.keys())
        for idx, left_group in enumerate(group_ids[:-1]):
            left_metrics = group_metrics[left_group]
            for right_group in group_ids[idx + 1 :]:
                right_metrics = group_metrics[right_group]
                left_tpr = left_metrics.get("TPR", float("nan"))
                right_tpr = right_metrics.get("TPR", float("nan"))
                left_fpr = left_metrics.get("FPR", float("nan"))
                right_fpr = right_metrics.get("FPR", float("nan"))
                if any(_is_nan(value) for value in (left_tpr, right_tpr, left_fpr, right_fpr)):
                    continue

                pair_eq_odds = max(abs(left_tpr - right_tpr), abs(left_fpr - right_fpr))
                if _is_nan(eq_odds) or pair_eq_odds > eq_odds:
                    eq_odds = pair_eq_odds
                    eq_odds_pair = (left_group, right_group)

                pair_avg_odds = 0.5 * (
                    abs(left_tpr - right_tpr) + abs(left_fpr - right_fpr)
                )
                if _is_nan(avg_odds) or pair_avg_odds > avg_odds:
                    avg_odds = pair_avg_odds
                    avg_odds_pair = (left_group, right_group)

        return {
            "protected_value": self.protected_value,
            "demographic_parity_diff": dp_diff,
            "demographic_parity_pair": dp_pair,
            "demographic_parity_ratio": dp_ratio,
            "demographic_parity_ratio_pair": dp_ratio_pair,
            "equal_opportunity_diff": tpr_diff,
            "equal_opportunity_pair": tpr_pair,
            "fpr_diff": fpr_diff,
            "fpr_diff_pair": fpr_pair,
            "equalized_odds_diff": eq_odds,
            "equalized_odds_pair": eq_odds_pair,
            "average_odds_diff": avg_odds,
            "average_odds_pair": avg_odds_pair,
            "accuracy_diff": acc_diff,
            "accuracy_diff_pair": acc_pair,
            "worst_group_id": worst_group_id,
            "worst_group_accuracy": worst_group_accuracy,
            "worst_group_error_rate": (
                float("nan")
                if worst_group_accuracy != worst_group_accuracy
                else 1.0 - worst_group_accuracy
            ),
        }
