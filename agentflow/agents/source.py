from abc import abstractmethod
from dataclasses import dataclass
import tensorflow as tf

from agentflow.logging import WithLogging
from agentflow.source import Source


class AgentSource(Source, WithLogging):

    @tf.function
    @abstractmethod
    def act(self, state, mask=None, **kwargs):
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

    @tf.function
    @abstractmethod
    def pnorms(self):
        ...

    @abstractmethod
    def save_weights(self, filepath):
        ...

    @abstractmethod
    def set_weights(self, weights):
        ...

    @tf.function
    @abstractmethod
    def update(self, *args, **kwargs):
        ...

@dataclass
class DiscreteActionAgentSource(AgentSource):

    num_actions: int

