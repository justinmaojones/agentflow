from dataclasses import dataclass
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

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
            return np.random.choice(self.num_actions, size=[len(state)])
        else:
            action = self.source.act(state, mask, **kwargs)
            if isinstance(action, tf.Tensor):
                action = action.numpy()
            return action

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



    unittest.main()


