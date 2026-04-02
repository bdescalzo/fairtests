import argparse
import gc
import os
import sys

import numpy as np
import torch
from folktables import ACSDataSource, ACSIncome

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from data_tools.preprocessing import prepare_fair_splits_from_chunks
from fairtests import get_hyperparams, run_fairtests
from examples.results_excel import write_hyperparams_xlsx, write_results_xlsx


FALLBACK_ALL_STATE_CODES = (
    "AL",
    "AK",
    "AZ",
    "AR",
    "CA",
    "CO",
    "CT",
    "DE",
    "FL",
    "GA",
    "HI",
    "ID",
    "IL",
    "IN",
    "IA",
    "KS",
    "KY",
    "LA",
    "ME",
    "MD",
    "MA",
    "MI",
    "MN",
    "MS",
    "MO",
    "MT",
    "NE",
    "NV",
    "NH",
    "NJ",
    "NM",
    "NY",
    "NC",
    "ND",
    "OH",
    "OK",
    "OR",
    "PA",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "UT",
    "VT",
    "VA",
    "WA",
    "WV",
    "WI",
    "WY",
)

try:
    from folktables.load_acs import state_list as _FOLKTABLES_STATE_LIST
except Exception:  # pragma: no cover - fallback when internals are unavailable
    _FOLKTABLES_STATE_LIST = None

SENSITIVE_FEATURE_INDEX = 9


def _resolve_all_state_codes():
    # folktables currently supports the 50 states + PR (no DC).
    if _FOLKTABLES_STATE_LIST is not None:
        return tuple(code for code in _FOLKTABLES_STATE_LIST if code != "PR")
    return FALLBACK_ALL_STATE_CODES


def _iter_state_arrays(data_source, states):
    total_states = len(states)
    for idx, state in enumerate(states, start=1):
        print(f"[Example] Loading state {state} ({idx}/{total_states})...", flush=True)
        acs_state = data_source.get_data(states=[state], download=True)
        if acs_state is None or len(acs_state) == 0:
            print(f"[Example] State {state} has no rows. Skipping.", flush=True)
            continue

        X, y, group = ACSIncome.df_to_numpy(acs_state)
        del acs_state
        del group

        yield X.astype(np.float32, copy=False), y.astype(np.float32, copy=False)
        gc.collect()


def main():
    parser = argparse.ArgumentParser(description="Folktables fairness example")
    parser.add_argument(
        "--min-k",
        type=int,
        default=1,
        help="Minimum samples per group to include in training.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of each class allocated to test split (0 < test_size < 1).",
    )
    args = parser.parse_args()
    if not (0.0 < args.test_size < 1.0):
        raise ValueError("--test-size must be strictly between 0 and 1.")

    print("[Example] Preparing all-states ACS data with streaming...", flush=True)
    data_source = ACSDataSource(
        survey_year=2018,
        horizon="1-Year",
        survey="person",
    )
    states = list(_resolve_all_state_codes())

    def chunk_factory():
        return _iter_state_arrays(data_source, states)

    prepared = prepare_fair_splits_from_chunks(
        chunk_factory=chunk_factory,
        protected_feature_index=SENSITIVE_FEATURE_INDEX,
        test_size=args.test_size,
        seed=42,
        min_train_group_size=args.min_k,
    )
    if args.min_k > 1:
        print(
            f"[Example] Filtered training groups with min_k={args.min_k}. "
            f"Kept {np.unique(prepared.g_train).size} groups.",
            flush=True,
        )

    print("[Example] Converting to tensors...", flush=True)
    X_train_t = torch.from_numpy(prepared.X_train)
    X_test_t = torch.from_numpy(prepared.X_test)
    X_train_full_t = torch.from_numpy(prepared.X_train_full)
    X_test_full_t = torch.from_numpy(prepared.X_test_full)
    y_train_t = torch.from_numpy(prepared.y_train)
    y_test_t = torch.from_numpy(prepared.y_test)
    g_train_t = torch.from_numpy(prepared.g_train)
    g_test_t = torch.from_numpy(prepared.g_test)
    del prepared
    gc.collect()

    base_model_class = None

    print("[Example] Running fairtests...", flush=True)
    results = run_fairtests(
        X_train_t,
        y_train_t,
        X_test_t,
        y_test_t,
        g_train_t,
        g_test_t,
        store_predictions=False,
        X_train_full=X_train_full_t,
        X_test_full=X_test_full_t,
        model_class=base_model_class,
    )
    hyperparams = get_hyperparams()

    output_path = write_results_xlsx(
        results=results,
        output_dir=os.path.dirname(__file__),
        results_name="folktables_results",
    )
    hyperparams_output_path = write_hyperparams_xlsx(
        hyperparams=hyperparams,
        output_dir=os.path.dirname(__file__),
        results_name="folktables_hyperparams",
    )
    print(f"[Example] Writing results to {output_path}", flush=True)
    print(
        f"[Example] Writing hyperparameters to {hyperparams_output_path}",
        flush=True,
    )

    print("[Example] Done.", flush=True)

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
