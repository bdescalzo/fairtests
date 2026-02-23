from fair_methods import Baseline, MetaLearning, Reptile
from metrics.metrics import StandardMetrics, FairnessMetrics


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
):
    print("[Fairtest] Starting evaluation pipeline.")
    if methods is None:
        methods = {
            "baseline": Baseline(),
            "maml": MetaLearning(),
            "reptile": Reptile(),
        }

    results = {}

    for name, method in methods.items():
        print(f"[Fairtest] Running method: {name}")
        method.load_data(X_train, y_train, X_test)
        method.fit(sensitive_train)

        if isinstance(method, MetaLearning) or isinstance(method, Reptile):
            y_prob = method.predict(sensitive_labels=sensitive_test)
        else:
            y_prob = method.predict()

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
            "y_prob": y_prob,
        }
        print(f"[Fairtest] Finished method: {name}")

    print("[Fairtest] All methods completed.")
    return results
