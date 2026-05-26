import numpy as np
from data_tools.preprocessing import prepare_fair_splits_from_arrays
from runner.experiments.base import Experiment

def generate_toy_dataset(n_samples=1000, seed=9845):
    rng = np.random.default_rng(seed)

    # Sensitive attribute: imbalanced, but assigned independently of X
    sensitive = rng.choice([0, 1], size=n_samples, p=[0.1, 0.9])

    # Same feature distribution for all sensitive groups
    features = rng.multivariate_normal(
        mean=[0, 0],
        cov=[[1, 0], [0, 1]],
        size=n_samples,
    )

    labels = np.zeros(n_samples, dtype=np.int64)

    x1 = features[:, 0]

    mask_s0 = sensitive == 0
    mask_s1 = sensitive == 1

    # Group 0:
    # If x1 > 0, Y is likely 1.
    # If x1 <= 0, Y is likely 0.
    mask_s0_pos = mask_s0 & (x1 > 0)
    labels[mask_s0_pos] = rng.choice(
        [0, 1],
        size=np.sum(mask_s0_pos),
        p=[0.2, 0.8],
    )

    mask_s0_neg = mask_s0 & (x1 <= 0)
    labels[mask_s0_neg] = rng.choice(
        [0, 1],
        size=np.sum(mask_s0_neg),
        p=[0.8, 0.2],
    )

    # Group 1:
    # Opposite rule.
    # If x1 > 0, Y is likely 0.
    # If x1 <= 0, Y is likely 1.
    mask_s1_pos = mask_s1 & (x1 > 0)
    labels[mask_s1_pos] = rng.choice(
        [0, 1],
        size=np.sum(mask_s1_pos),
        p=[0.8, 0.2],
    )

    mask_s1_neg = mask_s1 & (x1 <= 0)
    labels[mask_s1_neg] = rng.choice(
        [0, 1],
        size=np.sum(mask_s1_neg),
        p=[0.2, 0.8],
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
            X_val=prepared.X_val,
            y_val=prepared.y_val,
            sensitive_val=prepared.g_val,
            store_predictions=False,
            X_train_full=prepared.X_train_full,
            X_test_full=prepared.X_test_full,
            X_val_full=prepared.X_val_full,
            X_train_onehot=prepared.X_train_onehot,
            X_test_onehot=prepared.X_test_onehot,
            X_val_onehot=prepared.X_val_onehot,
            model_class=None,
        )

