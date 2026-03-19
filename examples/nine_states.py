import argparse
import gc
import os
import sys

import numpy as np
import torch
from folktables import ACSDataSource, ACSIncome
from sklearn.preprocessing import StandardScaler

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from fairtests import run_fairtests
from examples.results_excel import write_results_xlsx


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

        # Remove sensitive feature from the input (race is the last feature).
        X_no_race = np.delete(X, 9, axis=1).astype(np.float32, copy=False)
        y = y.astype(np.float32, copy=False)
        group = group.astype(np.int64, copy=False)
        del X

        yield X_no_race, y, group
        gc.collect()


def _build_stratified_test_mask(y, test_size, rng):
    y_int = y.astype(np.int8, copy=False)
    test_mask = np.zeros(y_int.shape[0], dtype=bool)
    for label in np.unique(y_int):
        label_idx = np.flatnonzero(y_int == label)
        if label_idx.size <= 1:
            continue
        n_test = int(round(test_size * label_idx.size))
        n_test = min(max(n_test, 1), label_idx.size - 1)
        if n_test > 0:
            selected = rng.choice(label_idx, size=n_test, replace=False)
            test_mask[selected] = True
    return test_mask


def _scale_chunk(X_chunk, mean, scale):
    X_chunk = X_chunk.astype(np.float32, copy=False)
    X_chunk -= mean
    X_chunk /= scale
    return X_chunk


def _prepare_all_states_dataset(data_source, test_size=0.2):
    states = ["AL", "AK", "AZ", "NH", "ME", "SD", "CA", "NY", "TX"]
    scaler = StandardScaler()

    print("[Example] Pass 1/2: fitting scaler on streamed training split...", flush=True)
    rng = np.random.default_rng(42)
    train_rows = 0
    test_rows = 0
    feature_dim = None

    for X_state, y_state, _ in _iter_state_arrays(data_source, states):
        if feature_dim is None:
            feature_dim = X_state.shape[1]
        elif X_state.shape[1] != feature_dim:
            raise RuntimeError(
                f"Inconsistent feature size across states: "
                f"expected {feature_dim}, got {X_state.shape[1]}"
            )

        test_mask = _build_stratified_test_mask(y_state, test_size=test_size, rng=rng)
        train_mask = ~test_mask

        if np.any(train_mask):
            scaler.partial_fit(X_state[train_mask])
            train_rows += int(train_mask.sum())
        if np.any(test_mask):
            test_rows += int(test_mask.sum())

        del X_state, y_state, test_mask, train_mask
        gc.collect()

    if feature_dim is None or train_rows == 0 or test_rows == 0:
        raise RuntimeError("Could not build non-empty train/test splits from all states.")

    mean = scaler.mean_.astype(np.float32, copy=False)
    scale = scaler.scale_.astype(np.float32, copy=False)
    scale[scale == 0.0] = 1.0

    print(
        f"[Example] Planned split sizes: train={train_rows:,}, test={test_rows:,}.",
        flush=True,
    )
    print("[Example] Pass 2/2: building scaled train/test arrays...", flush=True)

    X_train = np.empty((train_rows, feature_dim), dtype=np.float32)
    y_train = np.empty(train_rows, dtype=np.float32)
    g_train = np.empty(train_rows, dtype=np.int64)

    X_test = np.empty((test_rows, feature_dim), dtype=np.float32)
    y_test = np.empty(test_rows, dtype=np.float32)
    g_test = np.empty(test_rows, dtype=np.int64)

    rng = np.random.default_rng(42)
    train_ptr = 0
    test_ptr = 0

    for X_state, y_state, g_state in _iter_state_arrays(data_source, states):
        test_mask = _build_stratified_test_mask(y_state, test_size=test_size, rng=rng)
        train_mask = ~test_mask

        n_train = int(train_mask.sum())
        if n_train > 0:
            X_train_chunk = _scale_chunk(X_state[train_mask], mean, scale)
            X_train[train_ptr : train_ptr + n_train] = X_train_chunk
            y_train[train_ptr : train_ptr + n_train] = y_state[train_mask]
            g_train[train_ptr : train_ptr + n_train] = g_state[train_mask]
            train_ptr += n_train
            del X_train_chunk

        n_test = int(test_mask.sum())
        if n_test > 0:
            X_test_chunk = _scale_chunk(X_state[test_mask], mean, scale)
            X_test[test_ptr : test_ptr + n_test] = X_test_chunk
            y_test[test_ptr : test_ptr + n_test] = y_state[test_mask]
            g_test[test_ptr : test_ptr + n_test] = g_state[test_mask]
            test_ptr += n_test
            del X_test_chunk

        del X_state, y_state, g_state, test_mask, train_mask
        gc.collect()

    if train_ptr != train_rows or test_ptr != test_rows:
        raise RuntimeError(
            f"Split size mismatch after second pass "
            f"(train {train_ptr}/{train_rows}, test {test_ptr}/{test_rows})."
        )

    return X_train, y_train, X_test, y_test, g_train, g_test


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

    X_train, y_train, X_test, y_test, g_train, g_test = _prepare_all_states_dataset(
        data_source,
        test_size=args.test_size,
    )

    if args.min_k > 1:
        counts = np.bincount(g_train.astype(int))
        valid_groups = np.where(counts >= args.min_k)[0]
        valid_mask = np.isin(g_train, valid_groups)
        X_train = X_train[valid_mask]
        y_train = y_train[valid_mask]
        g_train = g_train[valid_mask]
        print(
            f"[Example] Filtered training groups with min_k={args.min_k}. "
            f"Kept {len(valid_groups)} groups.",
            flush=True,
        )
        del valid_mask

    protected_value = int(np.min(g_train))

    print("[Example] Converting to tensors...", flush=True)
    X_train_t = torch.from_numpy(X_train)
    X_test_t = torch.from_numpy(X_test)
    y_train_t = torch.from_numpy(y_train.astype(np.float32, copy=False))
    y_test_t = torch.from_numpy(y_test.astype(np.float32, copy=False))
    g_train_t = torch.from_numpy(g_train.astype(np.int64, copy=False))
    g_test_t = torch.from_numpy(g_test.astype(np.int64, copy=False))
    del X_train, X_test, y_train, y_test, g_train, g_test
    gc.collect()

    print("[Example] Running fairtests...", flush=True)
    results = run_fairtests(
        X_train_t,
        y_train_t,
        X_test_t,
        y_test_t,
        g_train_t,
        g_test_t,
        protected_value,
        store_predictions=False,
    )

    output_path = write_results_xlsx(
        results=results,
        output_dir=os.path.dirname(__file__),
        results_name="folktables_results",
    )
    print(f"[Example] Writing results to {output_path}", flush=True)

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
