import argparse
import os
import sys

import numpy as np
import torch
from folktables import ACSDataSource, ACSIncome
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from fairtests import run_fairtests
from fair_methods import Baseline, MetaLearning, Reptile
from examples.results_excel import write_results_xlsx


def main():
    parser = argparse.ArgumentParser(description="Folktables fairness example")
    parser.add_argument(
        "--min-k",
        type=int,
        default=2,
        help="Minimum samples per group to include in training.",
    )
    args = parser.parse_args()

    print("[Example] Loading ACS data...")
    data_source = ACSDataSource(
        survey_year=2018,
        horizon="1-Year",
        survey="person",
    )

    acs_data = data_source.get_data(
        states=["AL", "AK", "AZ", "NH", "ME", "SD", "CA", "NY", "TX"],
        download=True,
    )

    print("[Example] Preparing dataset...")
    X, y, group = ACSIncome.df_to_numpy(acs_data)

    # Remove sensitive feature from the input (race is the last feature)
    X_no_race = np.delete(X, 9, axis=1)

    X_train, X_test, y_train, y_test, g_train, g_test = train_test_split(
        X_no_race,
        y,
        group,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("[Example] Converting to tensors...")
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)
    g_train_t = torch.tensor(g_train, dtype=torch.long)
    g_test_t = torch.tensor(g_test, dtype=torch.long)

    protected_value = int(np.min(g_train))

    if args.min_k > 1:
        counts = np.bincount(g_train.astype(int))
        valid_groups = np.where(counts >= args.min_k)[0]
        valid_mask = np.isin(g_train, valid_groups)
        X_train_t = X_train_t[valid_mask]
        y_train_t = y_train_t[valid_mask]
        g_train_t = g_train_t[valid_mask]
        print(
            f"[Example] Filtered training groups with min_k={args.min_k}. "
            f"Kept {len(valid_groups)} groups."
        )

    print("[Example] Running fairtests...")
    methods = {
        "baseline": Baseline(),
        "maml": MetaLearning(),
        "reptile": Reptile(),
    }
    results = run_fairtests(
        X_train_t,
        y_train_t,
        X_test_t,
        y_test_t,
        g_train_t,
        g_test_t,
        protected_value,
        methods=methods,
    )

    output_path = write_results_xlsx(
        results=results,
        output_dir=os.path.dirname(__file__),
        results_name="folktables_results",
    )
    print(f"[Example] Writing results to {output_path}")

    print("[Example] Done.")

    for method_name, payload in results.items():
        print("=" * 72)
        print(f"Method: {method_name}")
        print("Overall metrics:")
        for k, v in payload["overall"].items():
            print(f"  {k}: {v}")

        print("\nMetrics by group:")
        for group_id, metrics in payload["by_group"].items():
            print(f"  Group {group_id}:")
            for k, v in metrics.items():
                print(f"    {k}: {v}")

        fairness = payload["fairness"]
        print("\nFairness metrics:")
        for k, v in fairness.items():
            if k.endswith("_metrics"):
                continue
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
