import numpy as np
import cv2
from typing import Tuple

class ImgEncoder:

    def __init__(self, key_to_encode: str, buffer_size: int, reshape: Tuple[int] = None):
        """
        Compresses elements within a dictionary at key=`key_to_encode` into PNG format.
        Assumes that the first dimension is a batch dimension.  When called, returns
        a dictionary with element at key=`key_to_encode` replaced by a numpy array
        with size `buffer_size`, excluding batch dim.

        key_to_encode: str
            Element of dictionary to compress
        buffer_size: int
            Buffer size for encodings
        rehsape: tuple[int]
            When not None, reshapes data at `key_to_encode`, excluding batch dim.
            Useful when data does not conform to standard image shapes.
        """

        assert isinstance(buffer_size, int) and buffer_size > 0, \
                "buffer_size must be a positive integer"
        if reshape is not None:
            assert isinstance(reshape, tuple), "reshape must be a tuple of ints"
            for x in reshape:
                assert isinstance(x, int), "reshape must be a tuple of ints"

        self._encoding_length_bytes = 4

        self.buffer_size = buffer_size
        self.key_to_encode = key_to_encode
        self.reshape = reshape

    def transform(self, data):
        # assume first dim is batch
        assert self.key_to_encode in data
        x = data[self.key_to_encode]
        if self.reshape:
            x = x.reshape(-1, *self.reshape)
        assert x.dtype == np.uint8, "data type must be uint8"
        assert x.ndim in (3, 4), "data must be 2d, or 3d, excluding batch dimension"
        n = len(x)
        encodings = []
        for i in range(n):
            e = cv2.imencode('.png', x[i])[1]
            assert e.ndim == 1 
            if len(e) <= self.buffer_size:
                encodings.append(e)
            else:
                raise ValueError(
                        f"could not append encoding of length={len(e)},"
                        f"because it is greater than max encoding size "
                        f"of {self.buffer_size}")

        m = len(encodings)

        # first 4 elements store length of encoding
        # length is stored as int subdivided into uint8
        encoded_array = np.zeros((m, self.buffer_size), dtype=np.uint8)

        # store encodings in arrary
        for i, e in enumerate(encodings):
            # store encoding
            encoded_array[i, :len(e)] = e

        output = {k: encoded_array if k==self.key_to_encode else data[k] for k in data}
        return output

    def __call__(self, data):
        return self.transform(data)

class ImgDecoder:

    def __init__(self, key_to_decode, reshape=None):
        """
        Decompresses elements within a dictionary at key=`key_to_encode` from PNG format.
        Assumes that the first dimension is a batch dimension.  When called, returns
        a dictionary with element at key=`key_to_encode` replaced by a decoded numpy array.

        key_to_encode: str
            Element of dictionary to compress
        rehsape: tuple[int]
            When not None, reshapes data at `key_to_encode`, excluding batch dim.
            Useful when data does not conform to standard image shapes.
        """

        if reshape is not None:
            assert isinstance(reshape, tuple), "reshape must be a tuple of ints"
            for x in reshape:
                assert isinstance(x, int), "reshape must be a tuple of ints"

        self.key_to_decode = key_to_decode
        self.reshape = reshape

    def transform(self, data):
        x_encoded = data[self.key_to_decode]
        assert x_encoded.dtype == np.uint8, "encoded data must be uint8"
        assert x_encoded.ndim == 2, "encoded data must be 1d, excluding batch dimension"
        n = len(x_encoded)

        decodings = []
        for i in range(n):
            decodings.append(cv2.imdecode(x_encoded[i], cv2.IMREAD_UNCHANGED))
        decoded_array = np.stack(decodings)
        output = {k: decoded_array if k==self.key_to_decode else data[k] for k in data}
        if self.reshape:
            output[self.key_to_decode] = output[self.key_to_decode].reshape(-1, *self.reshape)
        return output

    def __call__(self, data):
        return self.transform(data)
