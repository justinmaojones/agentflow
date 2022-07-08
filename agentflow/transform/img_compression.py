import numpy as np
import cv2

# for storing encoding length as uint8 and prefixing to encoded image array
_ENCODING_LENGTH_BYTES = 4

def _encode_to_uint8(x, partitions=_ENCODING_LENGTH_BYTES):
    x = np.array(x)
    p = np.arange(partitions)
    return np.right_shift(np.bitwise_and(x[..., None], 256**(p+1)-1), p*8).astype('uint8')

def _decode_from_uint8(x, partitions=_ENCODING_LENGTH_BYTES):
    p = np.arange(partitions)
    return np.sum(np.left_shift(x.astype('int64'), p*8), axis=-1)

class ImgEncoder:

    def __init__(self, key_to_encode, max_encoding_size):

        # trivial assertion
        assert isinstance(max_encoding_size, int)
        assert max_encoding_size <= 2**(8*_ENCODING_LENGTH_BYTES) - 1, \
                f"max_encoding_size={max_encoding_size} cannot be larger than _ENCODING_LENGTH_BYTES bits"

        self._encoding_length_bytes = 4

        self.max_encoding_size = max_encoding_size
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
            if len(e) <= self.max_encoding_size:
                encodings.append(e)
            else:
                print("WARNING: could not append encoding of length %d, because it is greater than max encoding size of %d" % (len(e), self.max_encoding_size))
        m = len(encodings)

        # first 4 elements store length of encoding
        # length is stored as int subdivided into uint8
        encoded_array = np.zeros((m, self.max_encoding_size+_ENCODING_LENGTH_BYTES), dtype=np.uint8)

        # encode and store lengths
        encoding_lengths = np.array(list(map(len, encodings)))
        encoded_array[:, :_ENCODING_LENGTH_BYTES] = _encode_to_uint8(encoding_lengths)

        # store encodings in arrary
        for i, e in enumerate(encodings):
            # store encoding
            encoded_array[i, _ENCODING_LENGTH_BYTES:len(e)+_ENCODING_LENGTH_BYTES] = e

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

        lengths = _decode_from_uint8(x_encoded[:, :_ENCODING_LENGTH_BYTES]) 

        decodings = []
        for i in range(n):
            left = _ENCODING_LENGTH_BYTES
            right = lengths[i] + _ENCODING_LENGTH_BYTES
            decodings.append(cv2.imdecode(x_encoded[i][left:right], cv2.IMREAD_UNCHANGED))
        decoded_array = np.stack(decodings)
        output = {k: decoded_array if k==self.key_to_decode else data[k] for k in data}
        return output

    def __call__(self, data):
        return self.transform(data)
