from abc import ABC, abstractmethod

class EvaluationMethod(ABC):
    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name

    @abstractmethod
    def __call__(self, model, device):
        pass


class PostEvaluationMethod(ABC):
    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name

    @abstractmethod
    def __call__(self):
        pass