import copy
from abc import ABC, abstractmethod

from fairtests import AVAILABLE_METHODS, get_hyperparams, run_fairtests


class Experiment(ABC):
    name = None

    def __init__(self, seed, method_names, hyperparams=None):
        if self.name is None:
            raise ValueError("Experiment subclasses must define a name.")

        self.seed = int(seed)
        self.method_names = self._normalize_method_names(method_names)
        self.hyperparams = self._normalize_hyperparams(hyperparams)

    @staticmethod
    def _normalize_method_names(method_names):
        if isinstance(method_names, str):
            names = [method_names]
        else:
            names = list(method_names)

        if not names:
            raise ValueError("method_names must contain at least one fair method.")

        names = list(dict.fromkeys(names))
        unknown = [name for name in names if name not in AVAILABLE_METHODS]
        if unknown:
            valid = ", ".join(sorted(AVAILABLE_METHODS))
            unknown_str = ", ".join(unknown)
            raise ValueError(
                f"Unknown method name(s): {unknown_str}. Valid names are: {valid}"
            )
        return names

    @staticmethod
    def _normalize_hyperparams(hyperparams):
        if hyperparams is None:
            return {}
        if not isinstance(hyperparams, dict):
            raise TypeError(
                "hyperparams must be a dict mapping method names to parameter dicts."
            )

        normalized = {}
        for method_name, params in hyperparams.items():
            if method_name not in AVAILABLE_METHODS:
                valid = ", ".join(sorted(AVAILABLE_METHODS))
                raise ValueError(
                    f"Unknown method in hyperparams: {method_name}. "
                    f"Valid names are: {valid}"
                )
            if not isinstance(params, dict):
                raise TypeError(
                    f"Hyperparameters for method '{method_name}' must be a dict."
                )
            normalized[method_name] = copy.deepcopy(params)
        return normalized

    def _build_methods(self):
        methods = {}
        for method_name in self.method_names:
            method_class = AVAILABLE_METHODS[method_name]
            method_hyperparams = copy.deepcopy(self.hyperparams.get(method_name, {}))

            def constructor(
                *, model_class=None, _method_class=method_class, _params=method_hyperparams
            ):
                return _method_class(model_class=model_class, **copy.deepcopy(_params))

            methods[method_name] = constructor
        return methods

    def _execute_fairtests(
        self,
        *,
        X_train,
        y_train,
        X_test,
        y_test,
        sensitive_train,
        sensitive_test,
        threshold=0.5,
        store_predictions=False,
        X_train_full=None,
        X_test_full=None,
        model_class=None,
    ):
        results = run_fairtests(
            X_train,
            y_train,
            X_test,
            y_test,
            sensitive_train,
            sensitive_test,
            threshold=threshold,
            methods=self._build_methods(),
            store_predictions=store_predictions,
            seed=self.seed,
            X_train_full=X_train_full,
            X_test_full=X_test_full,
            model_class=model_class,
        )
        return results, get_hyperparams()

    @abstractmethod
    def run(self):
        raise NotImplementedError
