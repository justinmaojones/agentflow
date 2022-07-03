from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from typing import Dict, List

from agentflow.buffers.nd_array_buffer import NDArrayBuffer
from agentflow.flow import Source


@dataclass
class BufferSource(Source):

    _buffers: Dict[str, NDArrayBuffer]

    def __getitem__(self, key: str):
        return self._buffers[key]

    @abstractmethod
    def __len__(self):
        ...

    @abstractmethod
    def append(self, data: Dict[str, np.ndarray]):
        ...

    @abstractmethod
    def get(self, time_idx: np.ndarray, batch_idx: np.ndarray = None):
        ...

    @abstractmethod
    def get_sequence(self, time_idx: np.ndarray, seq_size: int, batch_idx: np.ndarray = None):
        ...

    def extend(self, data_list: List[Dict[str, np.ndarray]]):
        for data in data_list:
            self.append(data)

    @abstractmethod
    def sample(self, n_samples: int):
        ...

    @property
    def shape(self):
        return {k:self._buffers[k].shape for k in self._buffers}

    @abstractmethod
    def tail(self, seq_size: int, batch_idx: np.ndarray = None):
        ...
