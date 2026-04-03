import gc

import numpy as np
from folktables import ACSDataSource, ACSIncome

from data_tools.preprocessing import prepare_fair_splits_from_chunks
from runner.experiments.base import Experiment


FALLBACK_ALL_STATE_CODES = (
    "AL",
    "AK",
    "AZ",
    "AR",
    "CA",
    "CO",
    "CT",
    "DE",
    "FL",
    "GA",
    "HI",
    "ID",
    "IL",
    "IN",
    "IA",
    "KS",
    "KY",
    "LA",
    "ME",
    "MD",
    "MA",
    "MI",
    "MN",
    "MS",
    "MO",
    "MT",
    "NE",
    "NV",
    "NH",
    "NJ",
    "NM",
    "NY",
    "NC",
    "ND",
    "OH",
    "OK",
    "OR",
    "PA",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "UT",
    "VT",
    "VA",
    "WA",
    "WV",
    "WI",
    "WY",
)

try:
    from folktables.load_acs import state_list as _FOLKTABLES_STATE_LIST
except Exception:  # pragma: no cover - fallback when internals are unavailable
    _FOLKTABLES_STATE_LIST = None


SENSITIVE_FEATURE_INDEX = 9


def resolve_all_state_codes():
    # folktables currently supports the 50 states + PR (no DC).
    if _FOLKTABLES_STATE_LIST is not None:
        return tuple(code for code in _FOLKTABLES_STATE_LIST if code != "PR")
    return FALLBACK_ALL_STATE_CODES


def iter_state_arrays(data_source, states):
    total_states = len(states)
    for idx, state in enumerate(states, start=1):
        print(
            f"[Experiment:{state}] Loading state {state} ({idx}/{total_states})...",
            flush=True,
        )
        acs_state = data_source.get_data(states=[state], download=True)
        if acs_state is None or len(acs_state) == 0:
            print(f"[Experiment:{state}] State {state} has no rows. Skipping.", flush=True)
            continue

        X, y, group = ACSIncome.df_to_numpy(acs_state)
        del acs_state
        del group

        yield X.astype(np.float32, copy=False), y.astype(np.float32, copy=False)
        gc.collect()


class FolktablesExperiment(Experiment):
    test_size = 0.2
    min_k = 1

    def __init__(
        self,
        seed,
        method_names,
        hyperparams=None,
        min_k=1,
        test_size=0.2,
    ):
        super().__init__(seed=seed, method_names=method_names, hyperparams=hyperparams)
        self.min_k = int(min_k)
        self.test_size = float(test_size)
        if not (0.0 < self.test_size < 1.0):
            raise ValueError("test_size must be strictly between 0 and 1.")

    def get_states(self):
        raise NotImplementedError

    def run(self):
        print(f"[Experiment:{self.name}] Preparing ACS data...", flush=True)
        data_source = ACSDataSource(
            survey_year=2018,
            horizon="1-Year",
            survey="person",
        )
        states = list(self.get_states())

        def chunk_factory():
            return iter_state_arrays(data_source, states)

        prepared = prepare_fair_splits_from_chunks(
            chunk_factory=chunk_factory,
            protected_feature_index=SENSITIVE_FEATURE_INDEX,
            test_size=self.test_size,
            seed=self.seed,
            min_train_group_size=self.min_k,
        )
        if self.min_k > 1:
            print(
                f"[Experiment:{self.name}] Filtered training groups with min_k={self.min_k}. "
                f"Kept {np.unique(prepared.g_train).size} groups.",
                flush=True,
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
