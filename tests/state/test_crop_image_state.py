import numpy as np
import unittest

from agentflow.state.crop_image_state import CropImageState


class TestCropImageState(unittest.TestCase):
    def test_update(self):

        state = CropImageState(top=1, bottom=1, left=1, right=1)
        x = np.ones((3, 4, 5, 6))
        y = state.update(x)
        np.testing.assert_array_equal(x[:, 1:-1, 1:-1, :], y)

        for i in [1, 2, 3, 5, 6, 7]:
            x = np.ones(tuple([4] * i))
            with self.assertRaises(ValueError):
                state.update(x)


if __name__ == "__main__":
    unittest.main()
