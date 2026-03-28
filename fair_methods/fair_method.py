from abc import ABC, abstractmethod


class FairMethod(ABC):
    def __init__(self, model_class=None, **kwargs):
        self.model_class = model_class

    @abstractmethod
    def load_data(self, X_train, y_train, X_test):
        raise NotImplementedError

    @abstractmethod
    def fit(self, sensitive_labels=None, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def predict(self, sensitive_labels=None, **kwargs):
        raise NotImplementedError
