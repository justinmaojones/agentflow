from abc import ABC, abstractmethod


class Schedule(ABC):
    @abstractmethod
    def __call__(self, step: int) -> float:
        ...
