import json
import math
import os
from numbers import Real

import numpy as np
import pandas as pd


def _normalize_excel_value(value):
    if isinstance(value, type):
        return value.__name__
    if hasattr(value, "item") and callable(value.item):
        try:
            value = value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, np.ndarray):
        return repr(value.tolist())
    if isinstance(value, (list, tuple, set, dict)):
        return repr(value)
    return value


def _to_json_ready(value):
    if isinstance(value, dict):
        return {str(key): _to_json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_ready(item) for item in value]
    if isinstance(value, set):
        return [_to_json_ready(item) for item in sorted(value, key=repr)]
    if isinstance(value, type):
        return value.__name__
    if isinstance(value, np.ndarray):
        return [_to_json_ready(item) for item in value.tolist()]
    if hasattr(value, "item") and callable(value.item):
        try:
            value = value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
    return value


def _is_numeric_scalar(value):
    if isinstance(value, bool):
        return False
    if hasattr(value, "item") and callable(value.item):
        try:
            value = value.item()
        except (TypeError, ValueError):
            return False
    return isinstance(value, Real)


def _numeric_summary(values):
    arr = np.asarray([float(value) for value in values], dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "count": int(arr.size),
    }


def _aggregate_metric_dict(metric_dicts):
    summary = {}
    metric_names = sorted(
        {metric_name for metric_dict in metric_dicts for metric_name in metric_dict}
    )
    for metric_name in metric_names:
        values = [
            metric_dict[metric_name]
            for metric_dict in metric_dicts
            if metric_name in metric_dict
        ]
        if values and all(_is_numeric_scalar(value) for value in values):
            summary[metric_name] = _numeric_summary(values)
        else:
            summary[metric_name] = {
                "values": [_to_json_ready(value) for value in values],
                "count": len(values),
            }
    return summary


def aggregate_results(seed_results):
    aggregated = {}
    method_names = sorted({method for results in seed_results for method in results})
    for method_name in method_names:
        method_payloads = [
            results[method_name] for results in seed_results if method_name in results
        ]
        by_group_payloads = [payload.get("by_group", {}) for payload in method_payloads]
        groups = sorted(
            {group_id for payload in by_group_payloads for group_id in payload},
            key=lambda group_id: repr(group_id),
        )

        aggregated[method_name] = {
            "overall": _aggregate_metric_dict(
                [payload.get("overall", {}) for payload in method_payloads]
            ),
            "fairness": _aggregate_metric_dict(
                [payload.get("fairness", {}) for payload in method_payloads]
            ),
            "by_group": {
                str(group_id): _aggregate_metric_dict(
                    [payload[group_id] for payload in by_group_payloads if group_id in payload]
                )
                for group_id in groups
            },
        }
    return aggregated


def write_json(data, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(_to_json_ready(data), f, indent=2, sort_keys=True)
    return output_path


def write_results_xlsx(results, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        overall_side = []
        fairness_side = []

        for method_name, payload in results.items():
            overall_row = {
                key: _normalize_excel_value(value)
                for key, value in payload["overall"].items()
            }
            overall_df = pd.DataFrame([overall_row])
            overall_df.insert(0, "method", method_name)
            overall_df.to_excel(writer, sheet_name=f"{method_name}_overall", index=False)

            by_group_rows = []
            for group_id, metrics in payload["by_group"].items():
                by_group_rows.append(
                    {
                        "group": _normalize_excel_value(group_id),
                        **{
                            key: _normalize_excel_value(value)
                            for key, value in metrics.items()
                        },
                    }
                )
            pd.DataFrame(by_group_rows).to_excel(
                writer,
                sheet_name=f"{method_name}_by_group",
                index=False,
            )

            fairness_row = {
                key: _normalize_excel_value(value)
                for key, value in payload["fairness"].items()
            }
            fairness_df = pd.DataFrame([fairness_row])
            fairness_df.insert(0, "method", method_name)
            fairness_df.to_excel(
                writer,
                sheet_name=f"{method_name}_fairness",
                index=False,
            )

            overall_side.append(overall_df)
            fairness_side.append(fairness_df)

        if overall_side:
            pd.concat(overall_side, ignore_index=True).to_excel(
                writer,
                sheet_name="summary_overall",
                index=False,
            )

        if fairness_side:
            pd.concat(fairness_side, ignore_index=True).to_excel(
                writer,
                sheet_name="summary_fairness",
                index=False,
            )

    return output_path


def write_hyperparams_xlsx(hyperparams, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    summary_rows = []
    long_rows = []
    for method_name, params in hyperparams.items():
        normalized_params = {
            name: _normalize_excel_value(value) for name, value in params.items()
        }
        summary_rows.append({"method": method_name, **normalized_params})
        for hyperparam_name, value in normalized_params.items():
            long_rows.append(
                {
                    "method": method_name,
                    "hyperparameter": hyperparam_name,
                    "value": value,
                }
            )

    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        pd.DataFrame(summary_rows).to_excel(
            writer,
            sheet_name="summary_hyperparams",
            index=False,
        )
        pd.DataFrame(long_rows).to_excel(
            writer,
            sheet_name="long_hyperparams",
            index=False,
        )

    return output_path


def write_summary_xlsx(summary_results, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    overall_rows = []
    fairness_rows = []
    by_group_rows = []

    for method_name, payload in summary_results.items():
        for metric_name, metric_summary in payload.get("overall", {}).items():
            overall_rows.append(
                {
                    "method": method_name,
                    "metric": metric_name,
                    "mean": metric_summary.get("mean"),
                    "std": metric_summary.get("std"),
                    "count": metric_summary.get("count"),
                    "values": _normalize_excel_value(metric_summary.get("values")),
                }
            )

        for metric_name, metric_summary in payload.get("fairness", {}).items():
            fairness_rows.append(
                {
                    "method": method_name,
                    "metric": metric_name,
                    "mean": metric_summary.get("mean"),
                    "std": metric_summary.get("std"),
                    "count": metric_summary.get("count"),
                    "values": _normalize_excel_value(metric_summary.get("values")),
                }
            )

        for group_id, metrics in payload.get("by_group", {}).items():
            for metric_name, metric_summary in metrics.items():
                by_group_rows.append(
                    {
                        "method": method_name,
                        "group": group_id,
                        "metric": metric_name,
                        "mean": metric_summary.get("mean"),
                        "std": metric_summary.get("std"),
                        "count": metric_summary.get("count"),
                        "values": _normalize_excel_value(metric_summary.get("values")),
                    }
                )

    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        pd.DataFrame(overall_rows).to_excel(
            writer, sheet_name="summary_overall", index=False
        )
        pd.DataFrame(fairness_rows).to_excel(
            writer, sheet_name="summary_fairness", index=False
        )
        pd.DataFrame(by_group_rows).to_excel(
            writer, sheet_name="summary_by_group", index=False
        )

    return output_path
