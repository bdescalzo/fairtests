import os
from datetime import datetime

import pandas as pd


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
            protected_metrics = fairness.pop("protected_group_metrics", {})
            non_protected_metrics = fairness.pop("non_protected_group_metrics", {})
            fairness_df = pd.DataFrame([fairness])
            fairness_df.insert(0, "method", method_name)
            fairness_df.to_excel(writer, sheet_name=f"{method_name}_fairness", index=False)

            overall_side.append(overall_df)
            fairness_side.append(fairness_df)

            if protected_metrics:
                prot_df = pd.DataFrame([protected_metrics])
                prot_df.insert(0, "method", method_name)
                prot_df.to_excel(
                    writer, sheet_name=f"{method_name}_protected", index=False
                )

            if non_protected_metrics:
                nonprot_df = pd.DataFrame([non_protected_metrics])
                nonprot_df.insert(0, "method", method_name)
                nonprot_df.to_excel(
                    writer, sheet_name=f"{method_name}_non_protected", index=False
                )

        if overall_side:
            overall_side_df = pd.concat(overall_side, ignore_index=True)
            overall_side_df.to_excel(writer, sheet_name="summary_overall", index=False)

        if fairness_side:
            fairness_side_df = pd.concat(fairness_side, ignore_index=True)
            fairness_side_df.to_excel(writer, sheet_name="summary_fairness", index=False)

    return output_path
