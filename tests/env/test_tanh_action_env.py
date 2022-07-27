import numpy as np
import unittest
from unittest.mock import patch

from agentflow.env.tanh_action_env import TanhActionEnv


class TestTanhActionEnv(unittest.TestCase):
    @patch("agentflow.env.source.EnvSource")
    def test_step(self, MockEnvSource):
        identity = lambda x: x
        MockEnvSource.step = identity
        env = TanhActionEnv(MockEnvSource, scale=1.1)

        x1 = 2.0
        x2 = np.array([-1, 0, 1], dtype=np.float32)
        x3 = np.array([[-1, 0, 1]], dtype=np.float32)

        y1 = env.step(x1)
        y2 = env.step(x2)
        y3 = env.step(x3)

        expected1 = 1.1 * np.tanh(x1)
        expected2 = 1.1 * np.tanh(x2)
        expected3 = 1.1 * np.tanh(x3)

        np.testing.assert_array_equal(expected1, y1)
        np.testing.assert_array_equal(expected2, y2)
        np.testing.assert_array_equal(expected3, y3)


if __name__ == "__main__":
    unittest.main()
