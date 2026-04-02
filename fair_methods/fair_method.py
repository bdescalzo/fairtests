import copy
from abc import ABC, abstractmethod


class FairMethod(ABC):
    def __init__(self, model_class=None, **kwargs):
        self.model_class = model_class
        self._hyperparams = {}

    def _set_hyperparams(self, **hyperparams):
        self._hyperparams = dict(hyperparams)

    def get_hyperparams(self):
        hyperparams = {}
        for name, value in self._hyperparams.items():
            hyperparams[name] = copy.deepcopy(getattr(self, name, value))
        hyperparams["model_class"] = self.model_class
        return hyperparams

    @abstractmethod
    def load_data(self, X_train, y_train, X_test):
        raise NotImplementedError

    @abstractmethod
    def fit(self, sensitive_labels=None, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def predict(self, sensitive_labels=None, **kwargs):
        raise NotImplementedError
