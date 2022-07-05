from dataclasses import dataclass
import tensorflow as tf
import tensorflow_probability as tfp

from agentflow.agents.flow import DiscreteActionAgentFlow

@dataclass
class EpsilonGreedy(DiscreteActionAgentFlow):

    epsilon: float = 0.05 

    def __post_init__(self):
        self._rng_explore = tfp.distributions.Bernoulli(
            probs=tf.constant([self.epsilon]), 
            dtype=tf.bool
        )
        self._rng_actions = tfp.distributions.Categorical(
            logits=tf.zeros(self.num_actions, dtype=tf.float32),
            dtype=tf.int32
        )

    def sample_actions(self, greedy_action: tf.Tensor):
        return tf.where(
            self._rng_explore.sample(greedy_action.shape)[..., 0],
            self._rng_actions.sample(greedy_action.shape)[..., 0],
            greedy_action,
        )

    @tf.function
    def act(self, state, mask=None, **kwargs):
        return self.sample_actions(self.source.act(state, mask, **kwargs))

if __name__ == '__main__':
    import numpy as np
    import unittest
    from unittest.mock import patch

    from agentflow.agents.source import AgentSource

    class Test(unittest.TestCase):

        @patch('agentflow.agents.source.DiscreteActionAgentSource')
        def test_sample_actions(self, MockAgentSource):
            eps_greedy = EpsilonGreedy(source=MockAgentSource, num_actions=3, epsilon=0.)
            greedy_actions = tf.constant([0,1,2,0,1,2], dtype=tf.int32)
            output = eps_greedy.sample_actions(greedy_actions)
            np.testing.assert_array_equal(greedy_actions.numpy(), output.numpy())

    unittest.main()


