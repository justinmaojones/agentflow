from dataclasses import dataclass
import tensorflow as tf
import tensorflow_probability as tfp

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

if __name__ == '__main__':
    import numpy as np
    import unittest
    from unittest.mock import patch
    from unittest.mock import MagicMock

    from agentflow.agents.source import AgentSource

    class Test(unittest.TestCase):

        @patch('agentflow.agents.source.DiscreteActionAgentSource')
        def test_act(self, MockAgentSource):
            MockAgentSource.num_actions = 3
            greedy_actions = tf.constant([0,1,2,0,1,2], dtype=tf.int64)
            MockAgentSource.act = MagicMock(return_value = greedy_actions)
            eps_greedy = EpsilonGreedy(source=MockAgentSource, epsilon=0.)
            state = tf.zeros(6)
            output = eps_greedy.act(state)
            np.testing.assert_array_equal(greedy_actions.numpy(), output)

        @patch('agentflow.agents.source.DiscreteActionAgentSource')
        def test_act_exp_decay_schedule(self, MockAgentSource):
            MockAgentSource.num_actions = 3
            greedy_actions = tf.constant([0,1,2,0,1,2], dtype=tf.int64)
            MockAgentSource.act = MagicMock(return_value = greedy_actions)
            from agentflow.numpy.schedules.exponential_decay import ExponentialDecaySchedule
            eps_schedule = ExponentialDecaySchedule(0.05, 0.0, 0.0)
            eps_greedy = EpsilonGreedy(source=MockAgentSource, epsilon=eps_schedule)
            state = tf.zeros(6)
            output = eps_greedy.act(state)
            output = eps_greedy.act(state)
            np.testing.assert_array_equal(greedy_actions.numpy(), output)

    unittest.main()


