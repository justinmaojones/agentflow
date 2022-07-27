import cv2
import numpy as np
import unittest

from agentflow.state.cvt_rgb2gray_image_state import CvtRGB2GrayImageState 

class TestCvtRGB2GrayImageState(unittest.TestCase):

    def test_update(self):

        state = CvtRGB2GrayImageState(flatten=False)
        state_flat = CvtRGB2GrayImageState(flatten=True)

        shape = (2, 3, 4, 3)
        x = np.arange(np.prod(shape)).reshape(*shape).astype('uint8')
        y = state.update(x)

        expected = np.zeros((2, 3, 4), dtype='uint8')
        expected[0] = cv2.cvtColor(x[0], cv2.COLOR_RGB2GRAY)
        expected[1] = cv2.cvtColor(x[1], cv2.COLOR_RGB2GRAY)
        expected = expected[:,:,:,None]

        np.testing.assert_array_equal(y, expected)

        y_flat = state_flat.update(x)

        expected_flat = np.zeros((2, 12), dtype='uint8')
        expected_flat[0] = cv2.cvtColor(x[0], cv2.COLOR_RGB2GRAY).ravel()
        expected_flat[1] = cv2.cvtColor(x[1], cv2.COLOR_RGB2GRAY).ravel()

        np.testing.assert_array_equal(y_flat, expected_flat)


if __name__ == '__main__':
    unittest.main()

