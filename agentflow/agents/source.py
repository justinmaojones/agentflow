from abc import abstractmethod
from dataclasses import dataclass
import tensorflow as tf

from agentflow.logging import WithLogging
from agentflow.source import Source


class AgentSource(Source, WithLogging):
    @abstractmethod
    def act(self, state, mask=None, explore=True, **kwargs):
        ...

    @abstractmethod
    def build_model(self):
        ...

    @abstractmethod
    def get_weights(self):
        ...

    @abstractmethod
    def load_weights(self, filepath):
        ...

    @abstractmethod
    def pnorms(self):
        ...

    @abstractmethod
    def save_weights(self, filepath):
        ...

    @abstractmethod
    def set_weights(self, weights):
        ...

    @abstractmethod
    def update(self, *args, **kwargs):
        ...


@dataclass
class DiscreteActionAgentSource(AgentSource):

    num_actions: int
