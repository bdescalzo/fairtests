import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from runner.experiments import AVAILABLE_EXPERIMENTS, get_experiment_class
from runner.result_io import (
    aggregate_results,
    write_curve_artifacts,
    write_hyperparams_xlsx,
    write_json,
    write_results_xlsx,
    write_summary_xlsx,
)


RESULTS_DIR = os.path.join(ROOT_DIR, "runner", "results")
DETERMINISTIC_SEED_SOURCE = 42


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run seeded experiment pipelines and collect aggregated results.",
        add_help=False,
    )
    parser.add_argument(
        "-s",
        "--seed-count",
        type=int,
        required=True,
        help="Number of seeds to execute.",
    )
    parser.add_argument(
        "-m",
        "--method",
        nargs="+",
        required=True,
        help="Fair method names to pass to fairtests.",
    )
    parser.add_argument(
        "-e",
        "--experiment",
        required=True,
        choices=sorted(AVAILABLE_EXPERIMENTS),
        help="Experiment name to run.",
    )
    parser.add_argument(
        "-d",
        "--deterministic",
        action="store_true",
        help="Generate the same seed list on every invocation.",
    )
    parser.add_argument(
        "-h",
        "--hyperparam",
        dest="hyperparam_file",
        help="JSON file with per-method hyperparameters.",
    )
    parser.add_argument(
        "--help",
        action="help",
        help="Show this help message and exit.",
    )
    return parser.parse_args()


def _load_hyperparams(hyperparam_file):
    if hyperparam_file is None:
        return {}

    with open(hyperparam_file, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise TypeError("Hyperparameter JSON must contain a top-level object.")
    return payload


def _generate_seeds(seed_count, deterministic):
    seed_count = int(seed_count)
    if seed_count <= 0:
        raise ValueError("seed-count must be > 0.")

    if deterministic:
        rng = np.random.default_rng(DETERMINISTIC_SEED_SOURCE)
    else:
        rng = np.random.default_rng()

    seeds = []
    seen = set()
    while len(seeds) < seed_count:
        seed = int(rng.integers(0, np.iinfo(np.int32).max))
        if seed in seen:
            continue
        seen.add(seed)
        seeds.append(seed)
    return seeds


def _build_run_dir(experiment_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(RESULTS_DIR, f"{experiment_name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=False)
    return run_dir


def main():
    args = _parse_args()
    hyperparams = _load_hyperparams(args.hyperparam_file)
    seeds = _generate_seeds(args.seed_count, args.deterministic)

    experiment_class = get_experiment_class(args.experiment)
    run_dir = _build_run_dir(args.experiment)

    print(f"[Orchestrator] Experiment: {args.experiment}", flush=True)
    print(f"[Orchestrator] Methods: {', '.join(args.method)}", flush=True)
    print(f"[Orchestrator] Seeds: {seeds}", flush=True)
    print(f"[Orchestrator] Results directory: {run_dir}", flush=True)

    seed_results = []

    for index, seed in enumerate(seeds, start=1):
        print(
            f"[Orchestrator] Running seed {seed} ({index}/{len(seeds)})...",
            flush=True,
        )
        seed_dir = os.path.join(run_dir, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=False)

        experiment = experiment_class(
            seed=seed,
            method_names=args.method,
            hyperparams=hyperparams,
        )
        results, used_hyperparams = experiment.run()
        seed_results.append(results)

        curve_artifacts = write_curve_artifacts(
            results, os.path.join(seed_dir, "plots")
        )
        write_results_xlsx(results, os.path.join(seed_dir, "results.xlsx"))
        write_hyperparams_xlsx(
            used_hyperparams, os.path.join(seed_dir, "hyperparams.xlsx")
        )
        write_json(
            {
                "experiment": args.experiment,
                "seed": seed,
                "methods": args.method,
                "results": results,
                "curve_artifacts": curve_artifacts,
            },
            os.path.join(seed_dir, "results.json"),
        )

    aggregated_results = aggregate_results(seed_results)
    write_summary_xlsx(aggregated_results, os.path.join(run_dir, "full_result.xlsx"))
    write_json(
        {
            "experiment": args.experiment,
            "methods": args.method,
            "deterministic": args.deterministic,
            "seed_source": (
                DETERMINISTIC_SEED_SOURCE if args.deterministic else "random"
            ),
            "seeds": seeds,
            "results": aggregated_results,
        },
        os.path.join(run_dir, "full_result.json"),
    )

    print("[Orchestrator] Completed all runs.", flush=True)


if __name__ == "__main__":
    main()
