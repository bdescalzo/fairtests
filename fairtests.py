import os
import random
import gc
import copy

import numpy as np
import torch

from fair_methods import Baseline, GroupDRO, MetaLearning, MinimaxParetoFairness, Reptile
from metrics.curves import compute_threshold_curve_bundle
from metrics.metrics import StandardMetrics, FairnessMetrics
from models import GenericModel

AVAILABLE_METHODS = {
    "baseline": Baseline,    
    "dro": GroupDRO,
    "maml": MetaLearning,
    "reptile": Reptile,
    "mmpf": MinimaxParetoFairness,
}

_LAST_HYPERPARAMS = {}


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


def _validate_input_variant(
    X_train,
    X_test,
    X_train_variant,
    X_test_variant,
    variant_name,
):
    has_train_variant = X_train_variant is not None
    has_test_variant = X_test_variant is not None
    if has_train_variant != has_test_variant:
        raise ValueError(
            f"X_train_{variant_name} and X_test_{variant_name} must either both be "
            "provided or both be omitted."
        )

    if not has_train_variant:
        return False

    if _num_rows(X_train_variant) != _num_rows(X_train):
        raise ValueError(
            f"X_train_{variant_name} must have the same number of rows as X_train."
        )
    if _num_rows(X_test_variant) != _num_rows(X_test):
        raise ValueError(
            f"X_test_{variant_name} must have the same number of rows as X_test."
        )

    return True


def _resolve_variant_method_name(method_name, existing_names, variant_name):
    if variant_name == "full":
        candidates = (
            f"{method_name}_full",
            f"{method_name}_full_input",
            f"{method_name}_protected",
        )
    elif variant_name == "one_hot":
        candidates = (
            f"{method_name}_one_hot",
            f"{method_name}_onehot",
            f"{method_name}_protected_one_hot",
        )
    else:
        raise ValueError(f"Unsupported input variant '{variant_name}'.")

    for candidate in candidates:
        if candidate not in existing_names:
            return candidate
    raise ValueError(
        f"Could not assign a unique results key for the {variant_name} variant of "
        f"'{method_name}'."
    )


def _validate_model_class(model_class):
    if model_class is not None and not isinstance(model_class, type):
        raise TypeError("model_class must be a class with the GenericModel interface.")


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
    threshold,
    store_predictions,
    seed,
    model_class,
):
    _set_global_determinism(seed)
    if isinstance(method_spec, type):
        constructor_kwargs = {}
        if model_class is not None:
            constructor_kwargs["model_class"] = model_class
        method = method_spec(**constructor_kwargs)
    elif callable(method_spec):
        method = method_spec(model_class=model_class)
    else:
        method = method_spec
        if model_class is not None:
            method.model_class = model_class
    if hasattr(method, "seed"):
        method.seed = seed
    method_hyperparams = (
        method.get_hyperparams() if hasattr(method, "get_hyperparams") else {}
    )
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
            threshold=threshold,
        ).compute()
        curves = compute_threshold_curve_bundle(
            y_test,
            y_prob,
            sensitive_test,
        )

        result = {
            "overall": overall_metrics,
            "by_group": group_metrics,
            "fairness": fairness,
            "curves": curves,
        }
        if store_predictions:
            result["y_prob"] = y_prob
        print(f"[Fairtest] Finished method: {method_name}")
        return result, method_hyperparams
    finally:
        del method
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def get_hyperparams():
    return copy.deepcopy(_LAST_HYPERPARAMS)


def run_fairtests(
    X_train,
    y_train,
    X_test,
    y_test,
    sensitive_train,
    sensitive_test,
    threshold=0.5,
    methods=None,
    method_names=None,
    store_predictions=True,
    seed=42,
    X_train_full=None,
    X_test_full=None,
    X_train_onehot=None,
    X_test_onehot=None,
    model_class=None,
):
    global _LAST_HYPERPARAMS
    print("[Fairtest] Starting evaluation pipeline.")
    _set_global_determinism(seed)
    _validate_model_class(model_class)
    if model_class is None:
        model_class = GenericModel
    _LAST_HYPERPARAMS = {}
    methods = _resolve_methods(methods=methods, method_names=method_names)
    has_full_inputs = _validate_input_variant(
        X_train,
        X_test,
        X_train_full,
        X_test_full,
        "full",
    )
    has_one_hot_inputs = _validate_input_variant(
        X_train,
        X_test,
        X_train_onehot,
        X_test_onehot,
        "one_hot",
    )

    results = {}
    full_method_names = {}
    one_hot_method_names = {}
    if has_full_inputs:
        existing_names = set(methods.keys())
        for name in methods:
            full_method_names[name] = _resolve_variant_method_name(
                name, existing_names, "full"
            )
            existing_names.add(full_method_names[name])
    else:
        existing_names = set(methods.keys())

    if has_one_hot_inputs:
        existing_names.update(full_method_names.values())
        for name in methods:
            one_hot_method_names[name] = _resolve_variant_method_name(
                name, existing_names, "one_hot"
            )
            existing_names.add(one_hot_method_names[name])

    for name, method_spec in methods.items():
        results[name], _LAST_HYPERPARAMS[name] = _run_single_method(
            method_name=name,
            method_spec=method_spec,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            sensitive_train=sensitive_train,
            sensitive_test=sensitive_test,
            threshold=threshold,
            store_predictions=store_predictions,
            seed=seed,
            model_class=model_class,
        )

        if has_full_inputs:
            full_method_name = full_method_names[name]
            (
                results[full_method_name],
                _LAST_HYPERPARAMS[full_method_name],
            ) = _run_single_method(
                method_name=full_method_name,
                method_spec=method_spec,
                X_train=X_train_full,
                y_train=y_train,
                X_test=X_test_full,
                y_test=y_test,
                sensitive_train=sensitive_train,
                sensitive_test=sensitive_test,
                threshold=threshold,
                store_predictions=store_predictions,
                seed=seed,
                model_class=model_class,
            )

        if has_one_hot_inputs:
            one_hot_method_name = one_hot_method_names[name]
            (
                results[one_hot_method_name],
                _LAST_HYPERPARAMS[one_hot_method_name],
            ) = _run_single_method(
                method_name=one_hot_method_name,
                method_spec=method_spec,
                X_train=X_train_onehot,
                y_train=y_train,
                X_test=X_test_onehot,
                y_test=y_test,
                sensitive_train=sensitive_train,
                sensitive_test=sensitive_test,
                threshold=threshold,
                store_predictions=store_predictions,
                seed=seed,
                model_class=model_class,
            )

    print("[Fairtest] All methods completed.")
    return results
