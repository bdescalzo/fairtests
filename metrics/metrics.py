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

        protected_mask = self.sensitive == self.protected_value
        non_protected_mask = self.sensitive != self.protected_value

        protected_metrics = StandardMetrics(
            y_true[protected_mask], y_prob[protected_mask], self.threshold
        ).compute()
        non_protected_metrics = StandardMetrics(
            y_true[non_protected_mask], y_prob[non_protected_mask], self.threshold
        ).compute()

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

        dp_diff = protected_metrics["selection_rate"] - non_protected_metrics["selection_rate"]
        dp_ratio = _safe_div(
            protected_metrics["selection_rate"],
            non_protected_metrics["selection_rate"],
            default=float("nan"),
        )
        tpr_diff = protected_metrics["TPR"] - non_protected_metrics["TPR"]
        fpr_diff = protected_metrics["FPR"] - non_protected_metrics["FPR"]
        acc_diff = protected_metrics["accuracy"] - non_protected_metrics["accuracy"]

        eq_odds = max(abs(tpr_diff), abs(fpr_diff))
        avg_odds = 0.5 * (abs(tpr_diff) + abs(fpr_diff))

        return {
            "protected_value": self.protected_value,
            "demographic_parity_diff": dp_diff,
            "demographic_parity_ratio": dp_ratio,
            "equal_opportunity_diff": tpr_diff,
            "fpr_diff": fpr_diff,
            "equalized_odds_diff": eq_odds,
            "average_odds_diff": avg_odds,
            "accuracy_diff": acc_diff,
            "worst_group_id": worst_group_id,
            "worst_group_accuracy": worst_group_accuracy,
            "worst_group_error_rate": (
                float("nan")
                if worst_group_accuracy != worst_group_accuracy
                else 1.0 - worst_group_accuracy
            ),
            "protected_group_metrics": protected_metrics,
            "non_protected_group_metrics": non_protected_metrics,
        }
