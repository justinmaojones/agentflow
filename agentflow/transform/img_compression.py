import numpy as np
import cv2

class ImgEncoder:

    def __init__(self, key_to_encode, buffer_size):

        # trivial assertion
        assert isinstance(buffer_size, int) and buffer_size > 0, \
                "buffer_size must be a positive integer"

        self._encoding_length_bytes = 4

        self.buffer_size = buffer_size
        self.key_to_encode = key_to_encode

    def transform(self, data):
        # assume first dim is batch
        assert self.key_to_encode in data
        x = data[self.key_to_encode]
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

    def __init__(self, key_to_decode):
        self.key_to_decode = key_to_decode

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
        return output

    def __call__(self, data):
        return self.transform(data)
