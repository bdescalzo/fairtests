from runner.experiments.folktables_common import FolktablesExperiment


class FolktablesNineStatesExperiment(FolktablesExperiment):
    name = "folktables_nine_states"

    def get_states(self):
        return ("AL", "AK", "AZ", "NH", "ME", "SD", "CA", "NY", "TX")
