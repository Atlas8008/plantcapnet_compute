from abc import ABC, abstractmethod


class InferenceMethod(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, model, device):
        pass


class PostInferenceMethod(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self):
        pass
