from dataclasses import dataclass
import numpy as np
import tensorflow as tf

from agentflow.agents.flow import DiscreteActionAgentFlow

@dataclass
class CompletelyRandomDiscreteUntil(DiscreteActionAgentFlow):
    """
    Hijacks the agent to produce completely random actions
    until a specified step has been reached
    """

    num_steps: int 

    def __post_init__(self):
        self._t = 0

    def act(self, state, mask=None, **kwargs):
        self._t += 1
        if self._t <= self.num_steps:
            if self.log is not None:
                self.log.append(f"{self.__class__.__name__}/random", 1)
            return np.random.choice(self.num_actions, size=[len(state)])
        else:
            if self.log is not None:
                self.log.append(f"{self.__class__.__name__}/random", 0)
            action = self.source.act(state, mask, **kwargs)
            if isinstance(action, tf.Tensor):
                action = action.numpy()
            return action
