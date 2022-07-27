import cv2
import numpy as np
import unittest

from agentflow.state.resize_image_state import ResizeImageState


class TestResizeImageState(unittest.TestCase):
    def test_update_3_channels(self):

        shape = (2, 3, 4, 3)
        resized_shape = (5, 4)  # note resized shape input to cv2 is reversed

        state = ResizeImageState(resized_shape, flatten=False)
        state_flat = ResizeImageState(resized_shape, flatten=True)

        x = np.arange(np.prod(shape)).reshape(*shape).astype("uint8")
        y = state.update(x)

        expected = np.zeros((2, 4, 5, 3), dtype="uint8")
        expected[0] = cv2.resize(x[0], resized_shape, interpolation=cv2.INTER_AREA)
        expected[1] = cv2.resize(x[1], resized_shape, interpolation=cv2.INTER_AREA)

        np.testing.assert_array_almost_equal_nulp(y, expected)

        y_flat = state_flat.update(x)
        expected_flat = np.zeros((2, 60), dtype="uint8")
        expected_flat[0] = cv2.resize(
            x[0], resized_shape, interpolation=cv2.INTER_AREA
        ).ravel()
        expected_flat[1] = cv2.resize(
            x[1], resized_shape, interpolation=cv2.INTER_AREA
        ).ravel()

        np.testing.assert_array_almost_equal_nulp(y_flat, expected_flat)

    def test_update_1_channel(self):

        shape = (2, 3, 4, 1)
        resized_shape = (5, 4)  # note resized shape input to cv2 is reversed

        state = ResizeImageState(resized_shape, flatten=False)
        state_flat = ResizeImageState(resized_shape, flatten=True)

        x = np.arange(np.prod(shape)).reshape(*shape).astype("uint8")
        y = state.update(x)

        expected = np.zeros((2, 4, 5), dtype="uint8")
        expected[0] = cv2.resize(x[0], resized_shape, interpolation=cv2.INTER_AREA)
        expected[1] = cv2.resize(x[1], resized_shape, interpolation=cv2.INTER_AREA)
        expected = expected[..., None]

        np.testing.assert_array_almost_equal_nulp(y, expected)

        y_flat = state_flat.update(x)
        expected_flat = np.zeros((2, 20), dtype="uint8")
        expected_flat[0] = cv2.resize(
            x[0], resized_shape, interpolation=cv2.INTER_AREA
        ).ravel()
        expected_flat[1] = cv2.resize(
            x[1], resized_shape, interpolation=cv2.INTER_AREA
        ).ravel()

        np.testing.assert_array_almost_equal_nulp(y_flat, expected_flat)


if __name__ == "__main__":
    unittest.main()
