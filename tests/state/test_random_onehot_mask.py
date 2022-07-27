import numpy as np
import unittest

from agentflow.state.random_onehot_mask import RandomOneHotMask

class TestRandomOneHotMask(unittest.TestCase):

    def test_reset(self):
        state = RandomOneHotMask(3)
        state.reset()
        assert state.state() is None

    def test_update_reset(self):
        state = RandomOneHotMask(3)
        state.reset()
        state.update(np.random.randn(5,6))
        np.testing.assert_array_equal(state.state().sum(axis=1), np.ones(5))
        state.update(np.random.randn(5,6))
        np.testing.assert_array_equal(state.state().sum(axis=1), np.ones(5))
        state.update(np.random.randn(5,6), np.array([0,1,1,0,1]))
        np.testing.assert_array_equal(state.state().sum(axis=1), np.ones(5))

if __name__ == '__main__':
    unittest.main()
