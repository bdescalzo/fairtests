import os
import sys

import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from data_tools.preprocessing import prepare_fair_splits_from_arrays
from fairtests import get_hyperparams, run_fairtests
from examples.results_excel import write_hyperparams_xlsx, write_results_xlsx


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


def main():
    print("[Toy Example] Generating toy dataset...")
    X, y, g = generate_toy_dataset(n_samples=100000, seed=9845)
    X_full = np.column_stack((X, g.astype(X.dtype, copy=False)))
    prepared = prepare_fair_splits_from_arrays(
        X_full=X_full,
        y=y,
        protected_feature_index=X_full.shape[1] - 1,
        test_size=0.2,
        seed=42,
    )

    print("[Toy Example] Converting to tensors...")
    X_train_t = prepared.X_train
    X_test_t = prepared.X_test
    X_train_full_t = prepared.X_train_full
    X_test_full_t = prepared.X_test_full
    y_train_t = prepared.y_train
    y_test_t = prepared.y_test
    g_train_t = prepared.g_train
    g_test_t = prepared.g_test

    base_model_class = None

    print("[Toy Example] Running fairtests...")
    results = run_fairtests(
        X_train_t,
        y_train_t,
        X_test_t,
        y_test_t,
        g_train_t,
        g_test_t,
        X_train_full=X_train_full_t,
        X_test_full=X_test_full_t,
        model_class=base_model_class,
    )
    hyperparams = get_hyperparams()

    output_path = write_results_xlsx(
        results=results,
        output_dir=os.path.dirname(__file__),
        results_name="toy_results",
    )
    hyperparams_output_path = write_hyperparams_xlsx(
        hyperparams=hyperparams,
        output_dir=os.path.dirname(__file__),
        results_name="toy_hyperparams",
    )
    print(f"[Toy Example] Writing results to {output_path}")
    print(f"[Toy Example] Writing hyperparameters to {hyperparams_output_path}")

    print("[Toy Example] Done.")

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
