import os
import sys

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from fairtests import run_fairtests
from examples.results_excel import write_results_xlsx


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


def _build_joint_stratify_labels(labels, sensitive):
    labels = np.asarray(labels, dtype=np.int64)
    _, encoded_sensitive = np.unique(
        np.asarray(sensitive, dtype=np.int64), return_inverse=True
    )
    _, encoded_labels = np.unique(labels, return_inverse=True)
    joint = encoded_sensitive * max(1, int(np.max(encoded_labels)) + 1) + encoded_labels

    counts = np.bincount(joint)
    if counts.size == 0 or np.any(counts < 2):
        return labels
    return joint


def main():
    print("[Toy Example] Generating toy dataset...")
    X, y, g = generate_toy_dataset(n_samples=100000, seed=9845)
    X_full = np.column_stack((X, g.astype(X.dtype, copy=False)))
    stratify_labels = _build_joint_stratify_labels(y, g)

    (
        X_train,
        X_test,
        X_train_full,
        X_test_full,
        y_train,
        y_test,
        g_train,
        g_test,
    ) = train_test_split(
        X,
        X_full,
        y,
        g,
        test_size=0.2,
        random_state=42,
        stratify=stratify_labels,
    )

    scaler = StandardScaler()
    scaler_full = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_train_full = scaler_full.fit_transform(X_train_full)
    X_test_full = scaler_full.transform(X_test_full)

    print("[Toy Example] Converting to tensors...")
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    X_train_full_t = torch.tensor(X_train_full, dtype=torch.float32)
    X_test_full_t = torch.tensor(X_test_full, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)
    g_train_t = torch.tensor(g_train, dtype=torch.long)
    g_test_t = torch.tensor(g_test, dtype=torch.long)

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

    output_path = write_results_xlsx(
        results=results,
        output_dir=os.path.dirname(__file__),
        results_name="toy_results",
    )
    print(f"[Toy Example] Writing results to {output_path}")

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
