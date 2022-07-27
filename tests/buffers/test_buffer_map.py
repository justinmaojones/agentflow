import numpy as np
import unittest

from agentflow.buffers import BufferMap


class TestBufferMap(unittest.TestCase):
    def test_creation(self):
        n = 10
        buf = BufferMap(n)
        buf.append(
            {
                "x": np.random.randn(2),
                "y": np.random.randn(2, 3),
                "z": np.random.randn(2, 3, 4),
            }
        )
        self.assertEqual(buf.shape, {"x": (1, 2), "y": (1, 2, 3), "z": (1, 2, 3, 4)})

    def test_append(self):
        n = 5
        buf = BufferMap(n)
        for i in range(10):
            buf.append({"x": i * np.ones(2), "y": i * np.ones((2, 3))})
            self.assertEqual(len(buf), min(i + 1, n))
            self.assertEqual(buf._index, (i + 1) % n)
        np.testing.assert_array_equal(
            buf._buffers["x"]._buffer, np.arange(5, 10)[:, None] * np.ones((1, 2))
        )
        np.testing.assert_array_equal(
            buf._buffers["y"]._buffer,
            np.arange(5, 10)[:, None, None] * np.ones((1, 2, 3)),
        )

        with self.assertRaises(TypeError):
            buf.append(1)

        with self.assertRaises(TypeError):
            buf.append({})

        with self.assertRaises(TypeError):
            buf.append({"x": np.ones(2)})

        with self.assertRaises(TypeError):
            BufferMap(n).append({"x": np.ones(2), "y": np.ones((3, 3))})

    def test_shape(self):
        n = 5
        buf = BufferMap(n)
        for i in range(10):
            buf.append(
                {
                    "x": np.random.randn(2),
                    "y": np.random.randn(2, 3),
                    "z": np.random.randn(2, 3, 4),
                }
            )
            m = min(i + 1, n)
            self.assertEqual(
                buf.shape, {"x": (m, 2), "y": (m, 2, 3), "z": (m, 2, 3, 4)}
            )

    def test_tail(self):
        n = 5
        buf = BufferMap(n)
        for i in range(10):
            buf.append(
                {
                    "x": np.random.randn(2),
                    "y": np.random.randn(2, 3),
                    "z": np.random.randn(2, 3, 4),
                }
            )
            if i > 1:
                tail = buf.tail(2)
                self.assertEqual(tail["x"].shape, (2, 2))
                self.assertEqual(tail["y"].shape, (2, 2, 3))
                self.assertEqual(tail["z"].shape, (2, 2, 3, 4))

    def test_get(self):
        n = 5
        buf = BufferMap(n)
        for i in range(10):
            buf.append(
                {
                    "x": np.random.randn(2),
                    "y": np.random.randn(2, 3),
                    "z": np.random.randn(2, 3, 4),
                }
            )
            if i > 1:
                output = buf.get(2)
                self.assertEqual(output["x"].shape, (2,))
                self.assertEqual(output["y"].shape, (2, 3))
                self.assertEqual(output["z"].shape, (2, 3, 4))

                output = buf.get(2, 1)
                self.assertEqual(output["x"].shape, ())
                self.assertEqual(output["y"].shape, (3,))
                self.assertEqual(output["z"].shape, (3, 4))

    def test_sample(self):
        n = 5
        buf = BufferMap(n)
        for i in range(10):
            buf.append(
                {
                    "x": np.random.randn(2),
                    "y": np.random.randn(2, 3),
                    "z": np.random.randn(2, 3, 4),
                }
            )
            if i > 1:
                sample = buf.sample(5)
                self.assertEqual(sample["x"].shape, (5,))
                self.assertEqual(sample["y"].shape, (5, 3))
                self.assertEqual(sample["z"].shape, (5, 3, 4))


if __name__ == "__main__":
    unittest.main()
