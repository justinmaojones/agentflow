import numpy as np
import unittest

from agentflow.buffers import NDArrayBuffer


class Test(unittest.TestCase):
    def test_tail_slices(self):
        n = 10
        buf = NDArrayBuffer(n)
        for i in range(3):
            buf.append(np.array([i]))

        self.assertEqual(buf._tail_slices(2), [slice(1, 3)])
        self.assertEqual(buf._tail_slices(3), [slice(0, 3)])

        with self.assertRaises(ValueError):
            buf._tail_slices(4)

        for i in range(3, 13):
            buf.append(np.array([i]))

        self.assertEqual(buf._index, 3)
        self.assertEqual(buf._tail_slices(2), [slice(1, 3)])
        self.assertEqual(buf._tail_slices(3), [slice(0, 3)])
        self.assertEqual(buf._tail_slices(4), [slice(9, 10), slice(0, 3)])
        self.assertEqual(buf._tail_slices(10), [slice(3, 10), slice(0, 3)])

    def test_tail(self):
        n = 3
        buf = NDArrayBuffer(n)
        buf.append(np.array([1, 2]))
        buf.append(np.array([3, 4]))

        # test tail prior to filling up buffer
        np.testing.assert_array_equal(buf.tail(1), np.array([[3, 4]]))
        np.testing.assert_array_equal(buf.tail(2), np.array([[1, 2], [3, 4]]))

        buf.append(np.array([5, 6]))

        # test tail when buffer is full
        np.testing.assert_array_equal(buf.tail(2), np.array([[3, 4], [5, 6]]))

        buf.append(np.array([7, 8]))

        # test tail when buffer is full and has been overwritten
        np.testing.assert_array_equal(buf.tail(1), np.array([[7, 8]]))
        np.testing.assert_array_equal(buf.tail(2), np.array([[5, 6], [7, 8]]))
        np.testing.assert_array_equal(buf.tail(3), np.array([[3, 4], [5, 6], [7, 8]]))

        # test batch_idx
        np.testing.assert_array_equal(buf.tail(3, 0), np.array([3, 5, 7]))
        np.testing.assert_array_equal(buf.tail(3, [0]), np.array([[3], [5], [7]]))
        np.testing.assert_array_equal(
            buf.tail(3, [1, 0]), np.array([[4, 6, 8], [3, 5, 7]]).T
        )

    def test_append(self):
        n = 10
        buf = NDArrayBuffer(n)
        self.assertEqual(len(buf), 0)
        for i in range(2 * n - 2):
            x = np.arange(i, i + 6).reshape(1, 2, 3)
            buf.append(x)
            np.testing.assert_array_equal(buf._buffer[i % n], x)
            self.assertEqual(len(buf), min(n, i + 1))
            self.assertEqual(buf._index, (i + 1) % n)

        self.assertEqual(buf._buffer.shape, (n, 1, 2, 3))

    def test_get(self):
        n = 10
        buf = NDArrayBuffer(n)
        self.assertEqual(len(buf), 0)
        for i in range(2 * n - 2):
            x = np.arange(i, i + 6).reshape(2, 3)
            buf.append(x)
            np.testing.assert_array_equal(buf.get(min(i + 1, n) - 1), x)
            np.testing.assert_array_equal(buf.get(min(i + 1, n) - 1, 1), x[1])

        np.testing.assert_array_equal(buf.get([0, 2, 4]), buf._buffer[[8, 0, 2]])
        np.testing.assert_array_equal(
            buf.get([0, 2, 4], [1, 0, 0]), buf._buffer[[8, 0, 2], [1, 0, 0]]
        )

    def test_get_sequence(self):
        n = 10
        buf = NDArrayBuffer(n)
        self.assertEqual(len(buf), 0)
        for i in range(2 * n - 2):
            x = np.arange(i, i + 6).reshape(1, 2, 3)
            buf.append(x)

        self.assertEqual(
            buf.get_sequence_slices(0, 1), [slice(buf._index, buf._index + 1)]
        )
        self.assertEqual(
            buf.get_sequence_slices(1, 2), [slice(buf._index, buf._index + 2)]
        )
        self.assertEqual(
            buf.get_sequence_slices(1, 2), [slice(buf._index, buf._index + 2)]
        )
        self.assertEqual(buf.get_sequence_slices(2, 1), [slice(0, 1)])
        self.assertEqual(
            buf.get_sequence_slices(2, 2),
            [slice(buf._index + 1, buf._index + 2), slice(0, 1)],
        )
        self.assertEqual(
            buf.get_sequence_slices(3, 4),
            [slice(buf._index, buf._index + 2), slice(0, 2)],
        )
        self.assertEqual(buf.get_sequence_slices(3, 2), [slice(0, 2)])
        self.assertEqual(buf.get_sequence_slices(4, 2), [slice(1, 3)])

        self.assertEqual(
            buf.get_sequence_slices(n - 1, 1), [slice(buf._index - 1, buf._index)]
        )
        self.assertEqual(
            buf.get_sequence_slices(n - 1, 2), [slice(buf._index - 2, buf._index)]
        )

        expected = np.concatenate(
            [
                buf._buffer[buf._index: buf._index + 2, :, :, :],
                buf._buffer[:2, :, :, :],
            ],
            axis=0,
        )
        np.testing.assert_array_equal(buf.get_sequence(3, 4), expected)

        with self.assertRaises(ValueError):
            buf.get_sequence_slices(i=0, seq_size=0)
        with self.assertRaises(ValueError):
            buf.get_sequence_slices(i=0, seq_size=2)
        with self.assertRaises(ValueError):
            buf.get_sequence_slices(i=n, seq_size=1)

    def test_sample(self):
        n = 10
        buf = NDArrayBuffer(n)
        self.assertEqual(len(buf), 0)
        for i in range(2 * n - 2):
            x = np.arange(i, i + 6).reshape(1, 2, 3)
            buf.append(x)

        self.assertEqual(buf.sample(1).shape, (1, 2, 3))
        self.assertEqual(buf.sample(20).shape, (20, 2, 3))

        with self.assertRaises(ValueError):
            buf.sample(0)

        with self.assertRaises(ValueError):
            NDArrayBuffer(n).sample(1)

    def test_sample_sequence(self):
        n = 10
        buf = NDArrayBuffer(n)
        self.assertEqual(len(buf), 0)
        for i in range(2 * n - 2):
            x = np.arange(i, i + 6).reshape(1, 2, 3)
            buf.append(x)

        self.assertEqual(buf.sample_sequence(1, 1).shape, (1, 1, 2, 3))
        self.assertEqual(buf.sample_sequence(1, 2).shape, (1, 2, 2, 3))
        self.assertEqual(buf.sample_sequence(3, 1).shape, (3, 1, 2, 3))
        self.assertEqual(buf.sample_sequence(3, 4).shape, (3, 4, 2, 3))

        with self.assertRaises(ValueError):
            buf.sample_sequence(0)

        with self.assertRaises(ValueError):
            buf.sample_sequence(1, 0)

        with self.assertRaises(ValueError):
            buf.sample_sequence(1, 11)

    def test_append_sequence(self):
        n = 10
        buf = NDArrayBuffer(n)

        x = np.arange(3)[:, None]
        buf.append_sequence(x)
        np.testing.assert_array_equal(buf._buffer[:3], x)
        self.assertEqual(buf._index, 3)
        self.assertEqual(buf._n, 3)

        x = np.arange(3, 3 + 10)[:, None]
        buf.append_sequence(x)
        np.testing.assert_array_equal(
            buf._buffer, np.concatenate([x[7:], x[:7]], axis=0)
        )
        self.assertEqual(buf._index, 3)
        self.assertEqual(buf._n, 10)

        x = np.arange(13, 13 + 13)[:, None]
        buf.append_sequence(x)
        np.testing.assert_array_equal(
            buf._buffer, np.concatenate([x[13 - 6 :], x[3 : 13 - 6]], axis=0)
        )
        self.assertEqual(buf._index, 6)
        self.assertEqual(buf._n, 10)

    def test_shape(self):
        n = 10
        buf = NDArrayBuffer(n)

        with self.assertRaises(ValueError):
            buf.shape

        x = np.arange(3)[:, None]
        buf.append_sequence(x)
        self.assertEqual(buf.shape, (3, 1))

        x = np.arange(30)[:, None]
        buf.append_sequence(x)
        self.assertEqual(buf.shape, (10, 1))


if __name__ == "__main__":
    unittest.main()
