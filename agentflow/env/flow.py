from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from agentflow.env.source import EnvSource
from agentflow.flow import Flow
from agentflow.logging import LogsTFSummary
from agentflow.logging import WithLogging


@dataclass
class EnvFlow(Flow, WithLogging):

    source: Union[EnvSource, EnvFlow]

    def reset(self):
        return self.source.reset()

    def set_log(self, log: LogsTFSummary):
        super().set_log(log)
        self.source.set_log(log)

    def step(self, action):
        return self.source.step(action)
