from abc import abstractmethod
from dataclasses import dataclass

from agentflow.logging import WithLogging
from agentflow.source import Source


class EnvSource(Source, WithLogging):
    @abstractmethod
    def reset(self):
        ...

    @abstractmethod
    def step(self, action):
        ...
