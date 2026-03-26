import os
import random
import gc

import numpy as np
import torch

from fair_methods import Baseline, MetaLearning, MinimaxParetoFairness, Reptile
from metrics.metrics import StandardMetrics, FairnessMetrics

AVAILABLE_METHODS = {
    "baseline": Baseline,
    "maml": MetaLearning,
    "reptile": Reptile,
    "mmpf": MinimaxParetoFairness,
}


def _set_global_determinism(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False

    torch.use_deterministic_algorithms(True)


def _resolve_methods(methods=None, method_names=None):
    if methods is not None:
        if not isinstance(methods, dict):
            raise ValueError(
                "methods must be a dict mapping method names to classes or instances."
            )
        return methods

    if method_names is None:
        selected_names = list(AVAILABLE_METHODS.keys())
    else:
        if isinstance(method_names, str):
            selected_names = [method_names]
        else:
            selected_names = list(method_names)
        selected_names = list(dict.fromkeys(selected_names))

        unknown = [name for name in selected_names if name not in AVAILABLE_METHODS]
        if unknown:
            valid = ", ".join(AVAILABLE_METHODS.keys())
            unknown_str = ", ".join(unknown)
            raise ValueError(
                f"Unknown method name(s): {unknown_str}. "
                f"Valid names are: {valid}"
            )

    # Return classes so methods are instantiated lazily (one-at-a-time).
    return {name: AVAILABLE_METHODS[name] for name in selected_names}


def _num_rows(array_like):
    if hasattr(array_like, "shape"):
        return int(array_like.shape[0])
    return len(array_like)


def _validate_full_baseline_inputs(X_train, X_test, X_train_full, X_test_full):
    has_train_full = X_train_full is not None
    has_test_full = X_test_full is not None
    if has_train_full != has_test_full:
        raise ValueError(
            "X_train_full and X_test_full must either both be provided or both be omitted."
        )

    if not has_train_full:
        return False

    if _num_rows(X_train_full) != _num_rows(X_train):
        raise ValueError("X_train_full must have the same number of rows as X_train.")
    if _num_rows(X_test_full) != _num_rows(X_test):
        raise ValueError("X_test_full must have the same number of rows as X_test.")

    return True


def _is_baseline_method(method_name, method_spec):
    if method_name == "baseline":
        return True
    if isinstance(method_spec, type):
        return issubclass(method_spec, Baseline)
    return isinstance(method_spec, Baseline)


def _resolve_full_baseline_name(existing_names):
    for candidate in ("baseline_full", "baseline_full_input", "baseline_protected"):
        if candidate not in existing_names:
            return candidate
    raise ValueError(
        "Could not assign a unique results key for the full-input baseline."
    )


def _run_single_method(
    *,
    method_name,
    method_spec,
    X_train,
    y_train,
    X_test,
    y_test,
    sensitive_train,
    sensitive_test,
    protected_value,
    threshold,
    store_predictions,
    seed,
):
    _set_global_determinism(seed)
    method = method_spec() if isinstance(method_spec, type) else method_spec
    if hasattr(method, "seed"):
        method.seed = seed
    try:
        print(f"[Fairtest] Running method: {method_name}")
        method.load_data(X_train, y_train, X_test)
        method.fit(sensitive_train)

        y_prob = method.predict(sensitive_labels=sensitive_test)

        standard = StandardMetrics(y_test, y_prob, threshold=threshold)
        group_metrics = standard.by_group(sensitive_test)
        overall_metrics = standard.compute()

        fairness = FairnessMetrics(
            y_test,
            y_prob,
            sensitive_test,
            protected_value,
            threshold=threshold,
        ).compute()

        result = {
            "overall": overall_metrics,
            "by_group": group_metrics,
            "fairness": fairness,
        }
        if store_predictions:
            result["y_prob"] = y_prob
        print(f"[Fairtest] Finished method: {method_name}")
        return result
    finally:
        del method
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_fairtests(
    X_train,
    y_train,
    X_test,
    y_test,
    sensitive_train,
    sensitive_test,
    protected_value,
    threshold=0.5,
    methods=None,
    method_names=None,
    store_predictions=True,
    seed=42,
    X_train_full=None,
    X_test_full=None,
):
    print("[Fairtest] Starting evaluation pipeline.")
    _set_global_determinism(seed)
    methods = _resolve_methods(methods=methods, method_names=method_names)
    has_full_baseline_inputs = _validate_full_baseline_inputs(
        X_train, X_test, X_train_full, X_test_full
    )

    results = {}
    baseline_method_names = [
        name for name, method_spec in methods.items() if _is_baseline_method(name, method_spec)
    ]
    if len(baseline_method_names) > 1 and has_full_baseline_inputs:
        raise ValueError(
            "Full-input baseline comparison is ambiguous when multiple baseline methods "
            "are selected."
        )

    full_baseline_name = None
    full_baseline_target = None
    if has_full_baseline_inputs and baseline_method_names:
        full_baseline_name = _resolve_full_baseline_name(set(methods.keys()))
        full_baseline_target = baseline_method_names[0]

    for name, method_spec in methods.items():
        results[name] = _run_single_method(
            method_name=name,
            method_spec=method_spec,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            sensitive_train=sensitive_train,
            sensitive_test=sensitive_test,
            protected_value=protected_value,
            threshold=threshold,
            store_predictions=store_predictions,
            seed=seed,
        )

        if full_baseline_name is not None and name == full_baseline_target:
            results[full_baseline_name] = _run_single_method(
                method_name=full_baseline_name,
                method_spec=method_spec,
                X_train=X_train_full,
                y_train=y_train,
                X_test=X_test_full,
                y_test=y_test,
                sensitive_train=sensitive_train,
                sensitive_test=sensitive_test,
                protected_value=protected_value,
                threshold=threshold,
                store_predictions=store_predictions,
                seed=seed,
            )

    print("[Fairtest] All methods completed.")
    return results
