from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
import numpy as np
from typing import Dict, Union

from agentflow.buffers.buffer_map import BufferMap
from agentflow.buffers.source import BufferSource
from agentflow.flow import Flow
from agentflow.logging import LogsTFSummary
from agentflow.logging import WithLogging


@dataclass
class BufferFlow(Flow, WithLogging):

    source: Union[BufferFlow, BufferSource]

    def __getitem__(self, key: str):
        return self.source[key]

    def __len__(self):
        return len(self.source)

    @abstractmethod
    def append(self, data: Dict[str, np.ndarray]):
        ...

    @abstractmethod
    def append_sequence(self, data: Dict[str, np.ndarray]):
        ...

    def get(self, time_idx: np.ndarray, batch_idx: np.ndarray = None):
        return self.source.get(time_idx, batch_idx)

    def get_sequence(
        self, time_idx: np.ndarray, seq_size: int, batch_idx: np.ndarray = None
    ):
        return self.source.get_sequence(time_idx, seq_size, batch_idx)

    def sample(self, n_samples: int, **kwargs):
        return self.source.sample(n_samples, **kwargs)

    @property
    def shape(self):
        return self.source.shape

    def tail(self, seq_size: int, batch_idx: np.ndarray = None):
        return self.source.tail(seq_size, batch_idx)

    def set_log(self, log: LogsTFSummary):
        super().set_log(log)
        self.source.set_log(log)
