from dataclasses import dataclass
import tensorflow as tf

from agentflow.numpy.ops import eps_greedy_noise
from agentflow.agents.flow import DiscreteActionAgentFlow

@dataclass
class EpsilonGreedy(DiscreteActionAgentFlow):

    epsilon: float 

    def __post_init__(self):
        self._t = 0

    def _get_eps(self):
        if isinstance(self.epsilon, float):
            return self.epsilon
        else:
            eps = self.epsilon(self._t)
            self._t += 1
            return eps

    def act(self, state, mask=None, **kwargs):
        greedy_action = self.source.act(state, mask, **kwargs)
        if isinstance(greedy_action, tf.Tensor):
            greedy_action = greedy_action.numpy()
        return eps_greedy_noise(greedy_action, self.num_actions, self._get_eps())
