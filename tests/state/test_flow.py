import numpy as np
import unittest
from unittest.mock import patch

from agentflow.state.flow import State
from agentflow.state.flow import StatefulEnvFlow


class TestStatefulEnvFlow(unittest.TestCase):
    @patch("agentflow.state.flow.State")
    @patch("agentflow.env.source.EnvSource")
    def test_reset(self, MockEnvSource, MockState):

        source_reset_output = {"state": np.zeros(3), "action": np.ones(3)}
        MockEnvSource.reset = lambda: source_reset_output
        MockState.update = lambda frame, mask=None: frame + 2

        env = StatefulEnvFlow(source=MockEnvSource, state=MockState)
        env_reset_output = env.reset()

        expected = {"state": np.zeros(3) + 2, "action": np.ones(3)}

        assert len(env_reset_output) == 2
        np.testing.assert_array_equal(env_reset_output["state"], expected["state"])
        np.testing.assert_array_equal(env_reset_output["action"], expected["action"])

    @patch("agentflow.state.flow.State")
    @patch("agentflow.env.source.EnvSource")
    def test_step(self, MockEnvSource, MockState):

        source_step_output = {"state": np.array([1, 1, 1]), "done": np.array([1, 2, 1])}
        MockEnvSource.step = lambda a: source_step_output
        MockState.update = (
            lambda frame, mask=None: frame if mask is None else frame + mask
        )

        env = StatefulEnvFlow(source=MockEnvSource, state=MockState)
        env_reset_output = env.step(1)

        expected = {"state": np.array([2, 3, 2]), "done": np.array([1, 2, 1])}

        assert len(env_reset_output) == 2
        np.testing.assert_array_equal(env_reset_output["state"], expected["state"])
        np.testing.assert_array_equal(env_reset_output["done"], expected["done"])


if __name__ == "__main__":
    unittest.main()
