import numpy as np

from data_tools.preprocessing import prepare_fair_splits_from_arrays
from runner.experiments.base import Experiment


def generate_toy_dataset(n_samples=1000, seed=9845):
    rng = np.random.default_rng(seed)

    sensitive = rng.choice([0, 1], size=n_samples, p=[0.1, 0.9]).astype(np.int64)
    labels = rng.integers(0, 2, size=n_samples, dtype=np.int64)
    features = np.zeros((n_samples, 2), dtype=np.float64)

    mask = (labels == 1) & (sensitive == 1)
    if np.any(mask):
        features[mask] = rng.multivariate_normal(
            mean=[6, 0], cov=[[1, 0], [0, 1]], size=np.sum(mask)
        )

    mask = (labels == 0) & (sensitive == 1)
    if np.any(mask):
        features[mask] = rng.multivariate_normal(
            mean=[2, 0], cov=[[1, 0], [0, 1]], size=np.sum(mask)
        )

    mask = (labels == 0) & (sensitive == 0)
    if np.any(mask):
        features[mask] = rng.multivariate_normal(
            mean=[-4, 2], cov=[[2.5, 0], [0, 2.5]], size=np.sum(mask)
        )

    mask = (labels == 1) & (sensitive == 0)
    if np.any(mask):
        features[mask] = rng.multivariate_normal(
            mean=[-2, 0], cov=[[2.5, 0], [0, 2.5]], size=np.sum(mask)
        )

    return features, labels, sensitive


class ExampleToyExperiment(Experiment):
    name = "example_toy"

    def __init__(
        self,
        seed,
        method_names,
        hyperparams=None,
        n_samples=100000,
        test_size=0.2,
    ):
        super().__init__(seed=seed, method_names=method_names, hyperparams=hyperparams)
        self.n_samples = int(n_samples)
        self.test_size = float(test_size)
        if self.n_samples <= 0:
            raise ValueError("n_samples must be > 0.")
        if not (0.0 < self.test_size < 1.0):
            raise ValueError("test_size must be strictly between 0 and 1.")

    def run(self):
        print(f"[Experiment:{self.name}] Generating toy dataset...", flush=True)
        X, y, g = generate_toy_dataset(n_samples=self.n_samples, seed=self.seed)
        X_full = np.column_stack((X, g.astype(X.dtype, copy=False)))

        prepared = prepare_fair_splits_from_arrays(
            X_full=X_full,
            y=y,
            protected_feature_index=X_full.shape[1] - 1,
            test_size=self.test_size,
            seed=self.seed,
        )

        print(f"[Experiment:{self.name}] Running fairtests...", flush=True)
        return self._execute_fairtests(
            X_train=prepared.X_train,
            y_train=prepared.y_train,
            X_test=prepared.X_test,
            y_test=prepared.y_test,
            sensitive_train=prepared.g_train,
            sensitive_test=prepared.g_test,
            store_predictions=False,
            X_train_full=prepared.X_train_full,
            X_test_full=prepared.X_test_full,
            model_class=None,
        )
