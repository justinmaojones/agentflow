import numpy as np
import tensorflow as tf
import unittest
from unittest.mock import patch
from unittest.mock import MagicMock

from agentflow.agents.source import AgentSource
from agentflow.agents import CompletelyRandomDiscreteUntil

class TestCompletelyRandomDiscreteUntil(unittest.TestCase):

    @patch('agentflow.agents.source.DiscreteActionAgentSource')
    def test_act(self, MockAgentSource):
        MockAgentSource.num_actions = 3
        greedy_actions = tf.constant([0, 1, 2], dtype=tf.int64)
        MockAgentSource.act = MagicMock(return_value = greedy_actions)
        crd = CompletelyRandomDiscreteUntil(source=MockAgentSource, num_steps=2)

        state = tf.constant([2, 3, 4], dtype=tf.float32)
        out0 = crd.act(state)
        out1 = crd.act(state)

        assert not MockAgentSource.act.called

        out2 = crd.act(state)
        out3 = crd.act(state)

        assert MockAgentSource.act.called

        np.testing.assert_array_equal(out2, greedy_actions.numpy()) 
        np.testing.assert_array_equal(out3, greedy_actions.numpy()) 

if __name__ == "__main__":
    unittest.main()
