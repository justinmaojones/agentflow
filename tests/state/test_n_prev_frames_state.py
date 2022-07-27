import numpy as np
import unittest

from agentflow.state.n_prev_frames_state import create_empty_state 
from agentflow.state.n_prev_frames_state import NPrevFramesState
from agentflow.state.n_prev_frames_state import shift_and_update_state 

class TestNPrevFramesState(unittest.TestCase):

    def test_create_empty_state(self):

        for dtype in [int, float]:
            for n in [3, 4]:
                for i in [2, 3, 4]:
                    output = create_empty_state(np.ones([1]*i).astype(dtype), n)
                    expected = np.zeros([1]*i + [n]).astype(dtype)
                    np.testing.assert_array_equal(output, expected)

    def test_shift_and_update_state(self):

        state = np.array(
            [[0, 1, 2, 3],
             [4, 5, 6, 7]]
        )
        frame = np.array([8, 9])
        expected = np.array(
            [[8, 0, 1, 2],
             [9, 4, 5, 6]]
        )
        output = shift_and_update_state(state, frame)

        np.testing.assert_array_equal(expected, output)

    def test_update(self):

        state = NPrevFramesState(n_prev_frames=4)

        shape = (2, 3)
        x = np.zeros(shape)

        y1 = state.update(x+1)
        y2 = state.update(x+2)
        y3 = state.update(x+3)
        y4 = state.update(x+4)

        expected1 = np.zeros((2,3,4))
        expected1[..., 0] = 1

        expected2 = np.zeros((2,3,4))
        expected2[..., 0] = 2
        expected2[..., 1] = 1

        expected3 = np.zeros((2,3,4))
        expected3[..., 0] = 3
        expected3[..., 1] = 2
        expected3[..., 2] = 1

        expected4 = np.zeros((2,3,4))
        expected4[..., 0] = 4
        expected4[..., 1] = 3
        expected4[..., 2] = 2
        expected4[..., 3] = 1

        np.testing.assert_array_equal(y1, expected1)
        np.testing.assert_array_equal(y2, expected2)
        np.testing.assert_array_equal(y3, expected3)
        np.testing.assert_array_equal(y4, expected4)

    def test_update_with_mask(self):

        state = NPrevFramesState(n_prev_frames=3)

        x = np.zeros(2)

        y1 = state.update(x+1)
        y2 = state.update(x+2, np.array([0, 1]))
        y3 = state.update(x+3, np.array([1, 0]))

        expected1 = np.array(
            [[1, 0, 0],
             [1, 0, 0]]
        )
        expected2 = np.array(
            [[2, 1, 0],
             [2, 0, 0]]
        )
        expected3 = np.array(
            [[3, 0, 0],
             [3, 2, 0]]
        )

        np.testing.assert_array_equal(y1, expected1)
        np.testing.assert_array_equal(y2, expected2)
        np.testing.assert_array_equal(y3, expected3)

    def test_update_flattened(self):

        state = NPrevFramesState(n_prev_frames=4, flatten=True)

        shape = (2, 3)
        x = np.zeros(shape)

        y1 = state.update(x+1)
        y2 = state.update(x+2)
        y3 = state.update(x+3)
        y4 = state.update(x+4)

        expected1 = np.zeros((2, 3, 4))
        expected1[..., 0] = 1

        expected2 = np.zeros((2, 3, 4))
        expected2[..., 0] = 2
        expected2[..., 1] = 1

        expected3 = np.zeros((2, 3, 4))
        expected3[..., 0] = 3
        expected3[..., 1] = 2
        expected3[..., 2] = 1

        expected4 = np.zeros((2, 3, 4))
        expected4[..., 0] = 4
        expected4[..., 1] = 3
        expected4[..., 2] = 2
        expected4[..., 3] = 1

        np.testing.assert_array_equal(y1, expected1.reshape(2, 12))
        np.testing.assert_array_equal(y2, expected2.reshape(2, 12))
        np.testing.assert_array_equal(y3, expected3.reshape(2, 12))
        np.testing.assert_array_equal(y4, expected4.reshape(2, 12))


if __name__ == '__main__':
    unittest.main()

