from abc import ABC, abstractmethod
from dataclasses import dataclass

from agentflow.env.flow import EnvFlow

@dataclass
class State(ABC):

    def __post_init__(self):
        self.reset()

    def reset(self):
        self._state = None 

    @abstractmethod
    def update(self, frame, reset_mask=None):
        ...

    def state(self):
        return self._state


@dataclass
class StatefulEnvFlow(EnvFlow):

    state: State

    def reset(self):
        prior_output = self.source.reset()
        self.state.reset()
        output = {k: prior_output[k] for k in prior_output if k != 'state'}
        output['state'] = self.state.update(prior_output['state'])
        return output

    def step(self, *args, **kwargs):
        prior_step_output = self.source.step(*args, **kwargs)
        # copy non-state data from previous step output
        output = {k: prior_step_output[k] for k in prior_step_output if k != 'state'}
        # update state
        output['state'] = self.state.update(prior_step_output['state'], prior_step_output['done'])
        return output
