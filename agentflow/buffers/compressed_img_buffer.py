from __future__ import annotations

import numpy as np
from typing import Dict, Union

from agentflow.buffers.flow import BufferFlow
from agentflow.buffers.source import BufferSource
from agentflow.transform import ImgDecoder
from agentflow.transform import ImgEncoder


class CompressedImageBuffer(BufferFlow):
    def __init__(
        self,
        source: Union[BufferFlow, BufferSource],
        encoding_buffer_size: int = 20000,
        keys_to_encode: list[str] = ["state", "state2"],
    ):

        self.source = source
        self._encoding_buffer_size = encoding_buffer_size
        self._keys_to_encode = keys_to_encode

        self._encoders = None
        self._decoders = None

    def _build_encoders_decoders(self, data: Dict[str, np.ndarray]):
        assert self._encoders is None and self._decoders is None
        _encoders = []
        _decoders = []
        for k in self._keys_to_encode:
            d = data[k]
            if d.ndim == 4 and d.shape[3] == 1:
                # correct for cv2 squeezing of last dim
                encoder = ImgEncoder(k, self._encoding_buffer_size)
                decoder = ImgDecoder(k, reshape=d.shape[1:])
            elif d.ndim > 4 or (d.ndim == 4 and (d.shape[3] not in [1, 3])):
                reshape = (d.shape[1], int(np.prod(d.shape[2:])))
                encoder = ImgEncoder(k, self._encoding_buffer_size, reshape=reshape)
                decoder = ImgDecoder(k, reshape=d.shape[1:])
            else:
                encoder = ImgEncoder(k, self._encoding_buffer_size)
                decoder = ImgDecoder(k)

            _encoders.append(encoder)
            _decoders.append(decoder)

        self._encoders = _encoders
        self._decoders = _decoders

    def _maybe_build_encoders_decoders(self, data: Dict[str, np.ndarray]):
        if self._encoders is None:
            assert self._decoders is None
            self._build_encoders_decoders(data)

    def encode(self, data: Dict[str, np.ndarray]):
        self._maybe_build_encoders_decoders(data)
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
            data[k] = data[k].reshape([s[k][0] * s[k][1], *s[k][2:]])

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

    def get_sequence(
        self, time_idx: np.ndarray, seq_size: int, batch_idx: np.ndarray = None
    ):
        raise NotImplementedError

    def sample(self, n_samples: int, **kwargs):
        return self.decode(self.source.sample(n_samples, **kwargs))

    def tail(self, seq_size: int, batch_idx: np.ndarray = None):
        return self._flattened_decode(self.source.tail(seq_size, batch_idx))
