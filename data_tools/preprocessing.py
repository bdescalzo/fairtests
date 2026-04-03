from dataclasses import dataclass

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler


@dataclass
class PreparedFairSplits:
    X_train: np.ndarray
    X_test: np.ndarray
    X_train_full: np.ndarray
    X_test_full: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    g_train: np.ndarray
    g_test: np.ndarray


def _to_torch_splits(prepared):
    return PreparedFairSplits(
        X_train=torch.from_numpy(prepared.X_train),
        X_test=torch.from_numpy(prepared.X_test),
        X_train_full=torch.from_numpy(prepared.X_train_full),
        X_test_full=torch.from_numpy(prepared.X_test_full),
        y_train=torch.from_numpy(prepared.y_train),
        y_test=torch.from_numpy(prepared.y_test),
        g_train=torch.from_numpy(prepared.g_train),
        g_test=torch.from_numpy(prepared.g_test),
    )


def _validate_test_size(test_size):
    test_size = float(test_size)
    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size must be strictly between 0 and 1.")
    return test_size


def _normalize_labels(y):
    y = np.asarray(y)
    if y.ndim != 1:
        y = y.reshape(-1)
    return y.astype(np.float32, copy=False)


def _normalize_feature_index(protected_feature_index, n_features):
    idx = int(protected_feature_index)
    if idx < 0:
        idx += n_features
    if idx < 0 or idx >= n_features:
        raise IndexError(
            f"protected_feature_index={protected_feature_index} is out of bounds "
            f"for input with {n_features} features."
        )
    return idx


def _extract_group_and_features(X_full, protected_feature_index):
    X_full = np.asarray(X_full)
    if X_full.ndim != 2:
        raise ValueError("X_full must be a 2D array.")

    protected_idx = _normalize_feature_index(
        protected_feature_index, X_full.shape[1]
    )
    X_full = X_full.astype(np.float32, copy=False)
    g = X_full[:, protected_idx].astype(np.int64, copy=False)
    X = np.delete(X_full, protected_idx, axis=1).astype(np.float32, copy=False)
    return X, X_full, g


def _build_group_label_stratified_test_mask(y, group, test_size, rng):
    y_int = np.asarray(y, dtype=np.int8)
    group_int = np.asarray(group, dtype=np.int64)
    test_mask = np.zeros(y_int.shape[0], dtype=bool)

    for group_id in np.unique(group_int):
        group_idx = np.flatnonzero(group_int == group_id)
        if group_idx.size <= 1:
            continue

        local_mask = np.zeros(group_idx.size, dtype=bool)
        y_group = y_int[group_idx]
        for label in np.unique(y_group):
            label_local_idx = np.flatnonzero(y_group == label)
            if label_local_idx.size <= 1:
                continue
            n_test = int(round(test_size * label_local_idx.size))
            n_test = min(max(n_test, 1), label_local_idx.size - 1)
            if n_test > 0:
                selected = rng.choice(label_local_idx, size=n_test, replace=False)
                local_mask[selected] = True

        if not np.any(local_mask):
            n_test = int(round(test_size * group_idx.size))
            n_test = min(max(n_test, 1), group_idx.size - 1)
            if n_test > 0:
                selected = rng.choice(group_idx.size, size=n_test, replace=False)
                local_mask[selected] = True

        test_mask[group_idx[local_mask]] = True

    return test_mask


def _fit_standard_scalers(X_train, X_train_full):
    scaler = StandardScaler()
    scaler_full = StandardScaler()
    scaler.fit(X_train)
    scaler_full.fit(X_train_full)
    return scaler, scaler_full


def _scale_chunk(X_chunk, mean, scale):
    X_chunk = X_chunk.astype(np.float32, copy=False)
    X_chunk -= mean
    X_chunk /= scale
    return X_chunk


def _filter_train_groups(prepared, min_train_group_size):
    min_train_group_size = int(min_train_group_size)
    if min_train_group_size <= 1:
        return prepared

    train_groups = np.asarray(prepared.g_train, dtype=np.int64)
    unique_groups, counts = np.unique(train_groups, return_counts=True)
    valid_groups = unique_groups[counts >= min_train_group_size]
    valid_mask = np.isin(train_groups, valid_groups)

    if not np.any(valid_mask):
        raise ValueError(
            "No training groups remain after applying "
            f"min_train_group_size={min_train_group_size}."
        )

    return PreparedFairSplits(
        X_train=prepared.X_train[valid_mask],
        X_test=prepared.X_test,
        X_train_full=prepared.X_train_full[valid_mask],
        X_test_full=prepared.X_test_full,
        y_train=prepared.y_train[valid_mask],
        y_test=prepared.y_test,
        g_train=prepared.g_train[valid_mask],
        g_test=prepared.g_test,
    )


def prepare_fair_splits_from_arrays(
    X_full,
    y,
    protected_feature_index,
    test_size=0.2,
    seed=42,
    min_train_group_size=1,
):
    test_size = _validate_test_size(test_size)
    X, X_full, g = _extract_group_and_features(X_full, protected_feature_index)
    y = _normalize_labels(y)

    n_samples = X_full.shape[0]
    if y.shape[0] != n_samples:
        raise ValueError("X_full and y must contain the same number of rows.")

    rng = np.random.default_rng(seed)
    test_mask = _build_group_label_stratified_test_mask(y, g, test_size, rng)
    train_mask = ~test_mask

    if not np.any(train_mask) or not np.any(test_mask):
        raise ValueError("Could not build non-empty train/test splits from the arrays.")

    X_train = X[train_mask]
    X_test = X[test_mask]
    X_train_full = X_full[train_mask]
    X_test_full = X_full[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]
    g_train = g[train_mask]
    g_test = g[test_mask]

    scaler, scaler_full = _fit_standard_scalers(X_train, X_train_full)
    X_train = scaler.transform(X_train).astype(np.float32, copy=False)
    X_test = scaler.transform(X_test).astype(np.float32, copy=False)
    X_train_full = scaler_full.transform(X_train_full).astype(np.float32, copy=False)
    X_test_full = scaler_full.transform(X_test_full).astype(np.float32, copy=False)

    prepared = PreparedFairSplits(
        X_train=X_train,
        X_test=X_test,
        X_train_full=X_train_full,
        X_test_full=X_test_full,
        y_train=y_train,
        y_test=y_test,
        g_train=g_train,
        g_test=g_test,
    )
    prepared = _filter_train_groups(prepared, min_train_group_size)
    return _to_torch_splits(prepared)


def prepare_fair_splits_from_chunks(
    chunk_factory,
    protected_feature_index,
    test_size=0.2,
    seed=42,
    min_train_group_size=1,
):
    if not callable(chunk_factory):
        raise TypeError(
            "chunk_factory must be callable and return a fresh iterable on each pass."
        )

    test_size = _validate_test_size(test_size)
    scaler = StandardScaler()
    scaler_full = StandardScaler()

    print("[Preprocessing] Pass 1/2: fitting scaler on streamed training split...")
    rng = np.random.default_rng(seed)
    train_rows = 0
    test_rows = 0
    feature_dim = None
    feature_dim_full = None

    for chunk_idx, (X_full_chunk, y_chunk) in enumerate(chunk_factory(), start=1):
        X_chunk, X_full_chunk, g_chunk = _extract_group_and_features(
            X_full_chunk, protected_feature_index
        )
        y_chunk = _normalize_labels(y_chunk)

        if X_chunk.shape[0] != y_chunk.shape[0]:
            raise ValueError(
                f"Chunk {chunk_idx} has mismatched feature/label lengths "
                f"({X_chunk.shape[0]} vs {y_chunk.shape[0]})."
            )

        if feature_dim is None:
            feature_dim = X_chunk.shape[1]
        elif X_chunk.shape[1] != feature_dim:
            raise RuntimeError(
                f"Inconsistent feature size across chunks: "
                f"expected {feature_dim}, got {X_chunk.shape[1]}."
            )

        if feature_dim_full is None:
            feature_dim_full = X_full_chunk.shape[1]
        elif X_full_chunk.shape[1] != feature_dim_full:
            raise RuntimeError(
                f"Inconsistent full feature size across chunks: "
                f"expected {feature_dim_full}, got {X_full_chunk.shape[1]}."
            )

        test_mask = _build_group_label_stratified_test_mask(
            y_chunk, g_chunk, test_size, rng
        )
        train_mask = ~test_mask

        if np.any(train_mask):
            scaler.partial_fit(X_chunk[train_mask])
            scaler_full.partial_fit(X_full_chunk[train_mask])
            train_rows += int(train_mask.sum())
        if np.any(test_mask):
            test_rows += int(test_mask.sum())

    if feature_dim is None or feature_dim_full is None or train_rows == 0 or test_rows == 0:
        raise RuntimeError("Could not build non-empty train/test splits from the chunks.")

    mean = scaler.mean_.astype(np.float32, copy=False)
    scale = scaler.scale_.astype(np.float32, copy=False)
    scale[scale == 0.0] = 1.0
    mean_full = scaler_full.mean_.astype(np.float32, copy=False)
    scale_full = scaler_full.scale_.astype(np.float32, copy=False)
    scale_full[scale_full == 0.0] = 1.0

    print(
        f"[Preprocessing] Planned split sizes: train={train_rows:,}, test={test_rows:,}."
    )
    print("[Preprocessing] Pass 2/2: building scaled train/test arrays...")

    X_train = np.empty((train_rows, feature_dim), dtype=np.float32)
    X_train_full = np.empty((train_rows, feature_dim_full), dtype=np.float32)
    y_train = np.empty(train_rows, dtype=np.float32)
    g_train = np.empty(train_rows, dtype=np.int64)

    X_test = np.empty((test_rows, feature_dim), dtype=np.float32)
    X_test_full = np.empty((test_rows, feature_dim_full), dtype=np.float32)
    y_test = np.empty(test_rows, dtype=np.float32)
    g_test = np.empty(test_rows, dtype=np.int64)

    rng = np.random.default_rng(seed)
    train_ptr = 0
    test_ptr = 0

    for chunk_idx, (X_full_chunk, y_chunk) in enumerate(chunk_factory(), start=1):
        X_chunk, X_full_chunk, g_chunk = _extract_group_and_features(
            X_full_chunk, protected_feature_index
        )
        y_chunk = _normalize_labels(y_chunk)

        if X_chunk.shape[0] != y_chunk.shape[0]:
            raise ValueError(
                f"Chunk {chunk_idx} has mismatched feature/label lengths "
                f"({X_chunk.shape[0]} vs {y_chunk.shape[0]})."
            )

        test_mask = _build_group_label_stratified_test_mask(
            y_chunk, g_chunk, test_size, rng
        )
        train_mask = ~test_mask

        n_train = int(train_mask.sum())
        if n_train > 0:
            X_train_chunk = _scale_chunk(X_chunk[train_mask], mean, scale)
            X_train_full_chunk = _scale_chunk(
                X_full_chunk[train_mask], mean_full, scale_full
            )
            X_train[train_ptr : train_ptr + n_train] = X_train_chunk
            X_train_full[train_ptr : train_ptr + n_train] = X_train_full_chunk
            y_train[train_ptr : train_ptr + n_train] = y_chunk[train_mask]
            g_train[train_ptr : train_ptr + n_train] = g_chunk[train_mask]
            train_ptr += n_train

        n_test = int(test_mask.sum())
        if n_test > 0:
            X_test_chunk = _scale_chunk(X_chunk[test_mask], mean, scale)
            X_test_full_chunk = _scale_chunk(
                X_full_chunk[test_mask], mean_full, scale_full
            )
            X_test[test_ptr : test_ptr + n_test] = X_test_chunk
            X_test_full[test_ptr : test_ptr + n_test] = X_test_full_chunk
            y_test[test_ptr : test_ptr + n_test] = y_chunk[test_mask]
            g_test[test_ptr : test_ptr + n_test] = g_chunk[test_mask]
            test_ptr += n_test

    if train_ptr != train_rows or test_ptr != test_rows:
        raise RuntimeError(
            f"Split size mismatch after second pass "
            f"(train {train_ptr}/{train_rows}, test {test_ptr}/{test_rows})."
        )

    prepared = PreparedFairSplits(
        X_train=X_train,
        X_test=X_test,
        X_train_full=X_train_full,
        X_test_full=X_test_full,
        y_train=y_train,
        y_test=y_test,
        g_train=g_train,
        g_test=g_test,
    )
    prepared = _filter_train_groups(prepared, min_train_group_size)
    return _to_torch_splits(prepared)
