from __future__ import annotations

import numpy as np
from typing import Dict, Union

from agentflow.buffers.flow import BufferFlow
from agentflow.buffers.source import BufferSource
from agentflow.transform import ImgDecoder
from agentflow.transform import ImgEncoder


class CompressedImageBuffer(BufferFlow):

    def __init__(self,
            source: Union[BufferFlow, BufferSource],
            max_encoding_size: int = 2000,
            keys_to_encode: list[str] = ['state', 'state2'],
        ):
        
        self.source = source
        self._max_encoding_size = max_encoding_size
        self._keys_to_encode = keys_to_encode

        self._encoders = [ImgEncoder(k, max_encoding_size) for k in keys_to_encode]
        self._decoders = [ImgDecoder(k) for k in keys_to_encode]

    def encode(self, data: Dict[str, np.ndarray]):
        for encoder in self._encoders:
            data = encoder(data)
        return data

    def decode(self, data: Dict[str, np.ndarray]):
        for decoder in self._decoders:
            data = decoder(data)
        return data

    def _flattened_decode(self, data: Dict[str, np.ndarray]):
        # flatten time and batch dim
        s = {}
        for k in self._keys_to_encode:
            s[k] = data[k].shape
            data[k] = data[k].reshape([s[k][0]*s[k][1], *s[k][2:]])

        # decode
        data = self.decode(data)

        # unflatten
        for k in self._keys_to_encode:
            data[k] = data[k].reshape(s[k][:2] + data[k].shape[1:])

        return data

    def append(self, data: Dict[str, np.ndarray]):
        self.source.append(self.encode(data))

    def append_sequence(self, data: Dict[str, np.ndarray]):
        raise NotImplementedError

    def get(self, time_idx: np.ndarray, batch_idx: np.ndarray = None):
        if batch_idx is None:
            return self._flattened_decode(self.source.get(time_idx))
        else:
            return self.decode(self.source.get(time_idx, batch_idx))

    def get_sequence(self, time_idx: np.ndarray, seq_size: int, batch_idx: np.ndarray = None):
        raise NotImplementedError

    def sample(self, n_samples: int, **kwargs):
        return self.decode(self.source.sample(n_samples, **kwargs))

    def tail(self, seq_size: int, batch_idx: np.ndarray = None):
        return self._flattened_decode(self.source.tail(seq_size, batch_idx))
