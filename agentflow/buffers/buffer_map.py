from dataclasses import dataclass
import numpy as np
from typing import Dict

from agentflow.buffers.nd_array_buffer import NDArrayBuffer
from agentflow.buffers.source import BufferSource


@dataclass
class BufferMap(BufferSource):
    def __init__(self, max_length: int = 1e6, **kwargs):
        super().__init__(_buffers={}, **kwargs)

        self.max_length = max_length
        self._n = 0
        self._index = 0

    def __len__(self):
        return self._n

    def append(self, data: Dict[str, np.ndarray]):

        if not (isinstance(data, dict) and len(data) > 0):
            raise TypeError("input to append must be a non-empty dictionary")

        if len(self._buffers) == 0:
            for k in data:
                self._buffers[k] = NDArrayBuffer(self.max_length)
                self._buffers[k].append(data[k])
            shape = self.shape
            if not (len(set([v[1] for v in shape.values()])) == 1):
                raise TypeError("batch dim of all buffer elements must be the same")

            self.first_dim_size = list(shape.values())[0][0]

        else:

            for k in self._buffers:
                if k not in data:
                    raise TypeError("data must contain all elements of buffer")
                self._buffers[k].append(data[k])

        self._index = (self._index + 1) % self.max_length
        self._n = min(self._n + 1, self.max_length)

    def append_sequence(self, data: Dict[str, np.ndarray]):

        shape = {k: data[k].shape for k in data}
        assert (
            len(set([(v[0], v[-1]) for v in shape.values()])) == 1
        ), "first and last dim of all data elements must be the same"
        shape_values = list(shape.values())[0]
        batch_size = shape_values[0]
        seq_size = shape_values[-1]

        if len(self._buffers) == 0:
            assert len(data) > 0
            for k in data:
                self._buffers[k] = NDArrayBuffer(self.max_length)
                self._buffers[k].append_sequence(data[k])

            self.first_dim_size = batch_size

        else:
            assert len(data) == len(
                self._buffers
            ), "data must contain all elements of buffer"
            for k in self._buffers:
                self._buffers[k].append_sequence(data[k])

        self._index = (self._index + seq_size) % self.max_length
        self._n = min(self._n + seq_size, self.max_length)

    def get(self, time_idx: np.ndarray, batch_idx: np.ndarray = None):
        return {k: self._buffers[k].get(time_idx, batch_idx) for k in self._buffers}

    def get_sequence(
        self, time_idx: np.ndarray, seq_size: int, batch_idx: np.ndarray = None
    ):
        return {
            k: self._buffers[k].get_sequence(time_idx, seq_size, batch_idx)
            for k in self._buffers
        }

    def sample(self, n_samples: int):
        idx_time = np.random.choice(self._n, size=n_samples, replace=True)
        idx_batch = np.random.choice(self.first_dim_size, size=n_samples, replace=True)
        output = {k: self._buffers[k].get(idx_time, idx_batch) for k in self._buffers}
        return output

    def tail(self, seq_size, batch_idx=None):
        return {k: self._buffers[k].tail(seq_size, batch_idx) for k in self._buffers}
