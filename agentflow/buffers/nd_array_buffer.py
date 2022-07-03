import numpy as np
import random

class NDArrayBuffer(object):
    
    def __init__(self, max_length=1e6):
        self._buffer = None
        self._index = 0
        self._n = 0
        self._max_length = max_length 
        
    def __len__(self):
        return self._n

    def __setitem__(self, idx, val):
        self._buffer[idx] = val

    def __getitem__(self, idx):
        return self._buffer[idx]

    @property
    def shape(self):
        if self._buffer is None:
            raise ValueError("buffer must have data before it can have a shape")
        return tuple([len(self)] + list(self._buffer.shape[1:]))

    @property
    def size(self):
        if self._buffer is None:
            raise ValueError("buffer must have data before it can have a shape")

        return np.prod(self.shape)

    def _build_buffer(self, shape, dtype):
        self._buffer = np.zeros(shape, dtype=dtype)
    
    def append(self, x):
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be of type np.ndarray")

        if self._buffer is None:
            # infer shape automatically
            shape = [self._max_length] + list(x.shape)
            self._build_buffer(shape, x.dtype)

        self._buffer[self._index] = x
        self._n = min(self._n+1, self._max_length)
        self._index = (self._index+1) % self._max_length

    def append_sequence(self, x):
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be of type np.ndarray")

        if self._buffer is None:
            # infer shape automatically
            shape = [self._max_length] + list(x.shape)[1:]
            self._build_buffer(shape, x.dtype)

        seq_size = x.shape[0] 
        i1 = self._index
        i2 = min(self._index + seq_size, self._max_length)
        segment_size = i2 - i1 
        self._buffer[i1:i2] = x[:segment_size]
        self._n = min(self._n + segment_size, self._max_length)
        self._index = (self._index + segment_size) % self._max_length
        if segment_size < seq_size:
            self.append_sequence(x[segment_size:])

    def extend(self, X):
        if isinstance(X, list):
            for x in X:
                self.append(x)
        elif isinstance(X, np.ndarray):
            self.append_sequence(X)
        else:
            raise TypeError("X must be a list or np.ndarray")

    def _tail_slices(self, seq_size):
        if not (seq_size <= len(self)):
            raise ValueError("seq_size must be less than or equal to len(self)")
        slices = []
        i1 = max(0, self._index - seq_size)
        i2 = self._index
        slices.append(slice(i1, i2))
        seq_size -= (i2-i1)
        if seq_size > 0:
            i1 = self._n - seq_size
            i2 = self._n
            slices.append(slice(i1, i2))
            if not (i2-i1 == seq_size):
                raise ValueError("there is an error in the logic, we should never get here")
        return list(reversed(slices))

    def tail(self, seq_size, batch_idx=None):
        slices = self._tail_slices(seq_size)
        if batch_idx is None:
            return np.concatenate([self._buffer[s] for s in slices], axis=0)
        else:
            return np.concatenate([self._buffer[s, batch_idx] for s in slices], axis=0)

    def get_sequence_slices(self, i, seq_size):
        # fetch sequence backwards, where i represents the last element in the sequence.
        # we do this because you typically want the history relative to an index.
        if not (seq_size > 0):
            raise ValueError("seq_size must be between 0 and len(self)")

        if not (i >= seq_size-1):
            raise ValueError("i must be greater than or equal to seq_size-1")

        if not (i < len(self)):
            raise ValueError("i must be less than len(self)")

        i_start = i - (seq_size - 1)

        j1 = (self._index + i_start) % self._max_length
        j2 = min(j1 + seq_size, self._max_length)
        cur_seq_length = j2 - j1

        seq_slices = [slice(j1, j2)]

        if cur_seq_length < seq_size:
            remaining_seq_length = seq_size - cur_seq_length
            seq_slices.append(slice(0, remaining_seq_length))

        return seq_slices

    def get_sequence(self, i, seq_size, batch_idx=None):
        slices = self.get_sequence_slices(i, seq_size)
        if batch_idx is None:
            return np.concatenate([self._buffer[s] for s in slices], axis=0)
        else:
            return np.concatenate([self._buffer[s, batch_idx] for s in slices], axis=0)

    def get(self, time_idx, batch_idx=None):
        time_idx = np.array(time_idx)
        if self._n == self._max_length:
            time_idx += self._index
        if batch_idx is None:
            return self._buffer[time_idx % self._n]
        else:
            return self._buffer[time_idx % self._n, batch_idx]

    def sample(self, n=1):
        if not (n > 0):
            raise ValueError("sample size must be greater than 0")

        if not (len(self) > 0):
            raise ValueError("cannot sample from buffer if it has no data appended")

        batch_size = self._buffer.shape[1]
        time_idx = np.random.randint(0, len(self), size=n)
        batch_idx = np.random.randint(0, batch_size, size=n)
        return self._buffer[time_idx, batch_idx]

    def sample_sequence(self, n=1, seq_size=1):
        if not (n > 0):
            raise ValueError("sample size must be greater than 0")

        if not (len(self) >= seq_size):
            raise ValueError("cannot sample from buffer if it is not as large as seq_size")

        if not (seq_size > 0):
            raise ValueError("seq_size must be greater than 0")

        batch_size = self._buffer.shape[1]
        output = []
        for i in range(n):
            time_idx = random.randint(seq_size, len(self)-1)
            batch_idx = random.randint(0, batch_size-1)
            output.append(self.get_sequence(time_idx, seq_size, batch_idx)[None])
        return np.concatenate(output, axis=0)


if __name__ == '__main__':
    import unittest

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
            np.testing.assert_array_equal(buf.tail(3, [1, 0]), np.array([[4, 6, 8], [3, 5, 7]]).T)

        def test_append(self):
            n = 10
            buf = NDArrayBuffer(n)
            self.assertEqual(len(buf), 0)
            for i in range(2*n-2):
                x = np.arange(i, i+6).reshape(1, 2, 3)
                buf.append(x)
                np.testing.assert_array_equal(buf._buffer[i % n], x)
                self.assertEqual(len(buf), min(n, i+1))
                self.assertEqual(buf._index, (i+1) % n)

            self.assertEqual(buf._buffer.shape, (n, 1, 2, 3))

        def test_get(self):
            n = 10
            buf = NDArrayBuffer(n)
            self.assertEqual(len(buf), 0)
            for i in range(2*n-2):
                x = np.arange(i, i+6).reshape(2, 3)
                buf.append(x)
                np.testing.assert_array_equal(buf.get(min(i+1, n)-1), x)
                np.testing.assert_array_equal(buf.get(min(i+1, n)-1, 1), x[1])

            np.testing.assert_array_equal(buf.get([0, 2, 4]), buf._buffer[[8, 0, 2]])
            np.testing.assert_array_equal(buf.get([0, 2, 4], [1, 0, 0]), buf._buffer[[8, 0, 2], [1, 0, 0]])


        def test_get_sequence(self):
            n = 10
            buf = NDArrayBuffer(n)
            self.assertEqual(len(buf), 0)
            for i in range(2*n-2):
                x = np.arange(i, i+6).reshape(1, 2, 3)
                buf.append(x)

            self.assertEqual(buf.get_sequence_slices(0, 1), [slice(buf._index, buf._index+1)])
            self.assertEqual(buf.get_sequence_slices(1, 2), [slice(buf._index, buf._index+2)])
            self.assertEqual(buf.get_sequence_slices(1, 2), [slice(buf._index, buf._index+2)])
            self.assertEqual(buf.get_sequence_slices(2, 1), [slice(0, 1)])
            self.assertEqual(buf.get_sequence_slices(2, 2), [slice(buf._index+1, buf._index+2), slice(0, 1)])
            self.assertEqual(buf.get_sequence_slices(3, 4), [slice(buf._index, buf._index+2), slice(0, 2)])
            self.assertEqual(buf.get_sequence_slices(3, 2), [slice(0, 2)])
            self.assertEqual(buf.get_sequence_slices(4, 2), [slice(1, 3)])

            self.assertEqual(buf.get_sequence_slices(n-1, 1), [slice(buf._index-1, buf._index)])
            self.assertEqual(buf.get_sequence_slices(n-1, 2), [slice(buf._index-2, buf._index)])

            expected = np.concatenate([
                    buf._buffer[buf._index:buf._index+2, :, :, :], 
                    buf._buffer[:2, :, :, :]
                ], axis=0)
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
            for i in range(2*n-2):
                x = np.arange(i, i+6).reshape(1, 2, 3)
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
            for i in range(2*n-2):
                x = np.arange(i, i+6).reshape(1, 2, 3)
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

            x = np.arange(3, 3+10)[:, None]
            buf.append_sequence(x)
            np.testing.assert_array_equal(buf._buffer, np.concatenate([x[7:], x[:7]], axis=0))
            self.assertEqual(buf._index, 3)
            self.assertEqual(buf._n, 10)

            x = np.arange(13, 13+13)[:, None]
            buf.append_sequence(x)
            np.testing.assert_array_equal(buf._buffer, np.concatenate([x[13-6:], x[3:13-6]], axis=0))
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

    unittest.main()
