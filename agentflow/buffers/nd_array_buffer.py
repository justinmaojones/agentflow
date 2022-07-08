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
