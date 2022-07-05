from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from agentflow.agents.source import AgentSource
from agentflow.agents.source import DiscreteActionAgentSource
from agentflow.flow import Flow

@dataclass
class AgentFlow(Flow):

    source: Union[AgentSource, AgentFlow]

    def act(self, state, mask=None, **kwargs):
        return self.source.act(state, mask, **kwargs)
        
    def get_weights(self):
        return self.source.get_weights()

    def load_weights(self, filepath):
        return self.source.load_weights(filepath)

    def pnorms(self):
        return self.source.pnorms()

    def save_weights(self, filepath):
        return self.source.save_weights(filepath)

    def set_weights(self, weights):
        return self.source.set_weights(weights)

    def update(self, *args, **kwargs):
        return self.source.update(*args, **kwargs)

@dataclass
class DiscreteActionAgentFlow(AgentFlow):

    source: Union[DiscreteActionAgentSource, DiscreteActionAgentFlow]

    @property
    def num_actions(self) -> int:
        return self.source.num_actions
