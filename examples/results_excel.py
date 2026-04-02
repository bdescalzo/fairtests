import os
from datetime import datetime

import pandas as pd


def _normalize_excel_value(value):
    if isinstance(value, type):
        return value.__name__
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, (list, tuple, set, dict)):
        return repr(value)
    return value


def write_results_xlsx(results, output_dir, results_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"{results_name}_{timestamp}.xlsx")

    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        overall_side = []
        fairness_side = []

        for method_name, payload in results.items():
            overall_df = pd.DataFrame([payload["overall"]])
            overall_df.insert(0, "method", method_name)
            overall_df.to_excel(writer, sheet_name=f"{method_name}_overall", index=False)

            by_group_df = pd.DataFrame.from_dict(payload["by_group"], orient="index")
            by_group_df.index.name = "group"
            by_group_df.reset_index().to_excel(
                writer, sheet_name=f"{method_name}_by_group", index=False
            )

            fairness = payload["fairness"].copy()
            fairness_df = pd.DataFrame([fairness])
            fairness_df.insert(0, "method", method_name)
            fairness_df.to_excel(writer, sheet_name=f"{method_name}_fairness", index=False)

            overall_side.append(overall_df)
            fairness_side.append(fairness_df)

        if overall_side:
            overall_side_df = pd.concat(overall_side, ignore_index=True)
            overall_side_df.to_excel(writer, sheet_name="summary_overall", index=False)

        if fairness_side:
            fairness_side_df = pd.concat(fairness_side, ignore_index=True)
            fairness_side_df.to_excel(writer, sheet_name="summary_fairness", index=False)

    return output_path


def write_hyperparams_xlsx(hyperparams, output_dir, results_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"{results_name}_{timestamp}.xlsx")

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
            writer, sheet_name="summary_hyperparams", index=False
        )
        pd.DataFrame(long_rows).to_excel(
            writer, sheet_name="long_hyperparams", index=False
        )

    return output_path
