import numpy as np
import tensorflow as tf
import unittest
from unittest.mock import patch
from unittest.mock import MagicMock

from agentflow.agents import EpsilonGreedy 

class TestEpsilonGreedy(unittest.TestCase):

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

if __name__ == '__main__':
    unittest.main()
