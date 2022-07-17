from dataclasses import dataclass
import tensorflow as tf
from typing import Union

from agentflow.agents.flow import DiscreteActionAgentFlow
from agentflow.logging import LogsTFSummary
from agentflow.numpy.ops import eps_greedy_noise
from agentflow.numpy.schedules.schedule import Schedule

@dataclass
class EpsilonGreedy(DiscreteActionAgentFlow):

    epsilon: Union[float, Schedule]

    def __post_init__(self):
        self._t = 0

    def _get_eps(self):
        if isinstance(self.epsilon, float):
            eps = self.epsilon
        elif isinstance(self.epsilon, Schedule):
            eps = self.epsilon(self._t)
            self._t += 1
        else:
            raise NotImplementedError(
                f"Unhandled type: {type(self.epsilon)}. epsilon must be a float or Schedule")
    
        if self.log is not None:
            self.log.append(f"{self.__class__.__name__}/epsilon", eps)

        return eps

    def act(self, state, mask=None, **kwargs):
        greedy_action = self.source.act(state, mask, **kwargs)
        if isinstance(greedy_action, tf.Tensor):
            greedy_action = greedy_action.numpy()
        return eps_greedy_noise(greedy_action, self.num_actions, self._get_eps())
