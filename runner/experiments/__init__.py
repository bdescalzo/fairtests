import importlib


AVAILABLE_EXPERIMENTS = {
    "folktables_full": "runner.experiments.folktables_full:FolktablesFullExperiment",
    "folktables_nine_states": (
        "runner.experiments.folktables_nine_states:FolktablesNineStatesExperiment"
    ),
    "example_toy": "runner.experiments.example_toy:ExampleToyExperiment",
}


def get_experiment_class(experiment_name):
    try:
        module_path, class_name = AVAILABLE_EXPERIMENTS[experiment_name].split(":", 1)
    except KeyError as exc:
        valid = ", ".join(sorted(AVAILABLE_EXPERIMENTS))
        raise ValueError(
            f"Unknown experiment '{experiment_name}'. Valid experiments are: {valid}"
        ) from exc

    module = importlib.import_module(module_path)
    return getattr(module, class_name)


__all__ = ["AVAILABLE_EXPERIMENTS", "get_experiment_class"]
