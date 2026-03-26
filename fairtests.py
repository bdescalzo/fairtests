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
):
    print("[Fairtest] Starting evaluation pipeline.")
    _set_global_determinism(seed)
    methods = _resolve_methods(methods=methods, method_names=method_names)

    results = {}

    for name, method_spec in methods.items():
        _set_global_determinism(seed)
        method = method_spec() if isinstance(method_spec, type) else method_spec
        if hasattr(method, "seed"):
            method.seed = seed
        try:
            print(f"[Fairtest] Running method: {name}")
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

            results[name] = {
                "overall": overall_metrics,
                "by_group": group_metrics,
                "fairness": fairness,
            }
            if store_predictions:
                results[name]["y_prob"] = y_prob
            print(f"[Fairtest] Finished method: {name}")
        finally:
            del method
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("[Fairtest] All methods completed.")
    return results
