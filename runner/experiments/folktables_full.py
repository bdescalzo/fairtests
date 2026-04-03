from runner.experiments.folktables_common import (
    FolktablesExperiment,
    resolve_all_state_codes,
)


class FolktablesFullExperiment(FolktablesExperiment):
    name = "folktables_full"

    def get_states(self):
        return resolve_all_state_codes()
