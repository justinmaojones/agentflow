import numpy as np
import cv2

# for storing encoding length as uint8 and prefixing to encoded image array
_ENCODING_LENGTH_BYTES = 4
_ENCODING_LENGTH_PARTITIONS = [(256**(b+1)-1) ^ (256**b-1) for b in range(_ENCODING_LENGTH_BYTES)]

def _encode_length(x):
    return [x & e for e in _ENCODING_LENGTH_PARTITIONS]

class ImgEncoder(object):

    def __init__(self, key_to_encode, max_encoding_size):

        # trivial assertion
        assert isinstance(max_encoding_size, int)
        assert max_encoding_size <= 2**32 - 1, \
                f"max_encoding_size={max_encoding_size} cannot be larger than int32 max value"

        self._encoding_length_bytes = 4

        self.max_encoding_size = max_encoding_size
        self.key_to_encode = key_to_encode

    def transform(self, data):
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
        # length is stored as int32 subdivided into uint8
        encoded_array = np.zeros((m, self.max_encoding_size+_ENCODING_LENGTH_BYTES), dtype=np.uint8)
        lengths = np.zeros(m, dtype=int)
        for i in range(m):
            e = encodings[i]
            # store encoding length
            encoded_array[i, :_ENCODING_LENGTH_BYTES] = _encode_length(len(e))
            # store encoding
            encoded_array[i, _ENCODING_LENGTH_BYTES:len(e)+_ENCODING_LENGTH_BYTES] = e

        output = {k: encoded_array if k==self.key_to_encode else data[k] for k in data}
        return output

    def __call__(self, data):
        return self.transform(data)

if __name__ == '__main__':
    import unittest

    class Test(unittest.TestCase):

        def test_normal(self):
            img_encoder = ImgEncoder('state', 70)
            x = {
                'state':  np.array([[[0]], [[1]]], dtype='uint8'), 
                'something_else': np.array([1,2,3])
            }
            x2 = img_encoder(x)
            np.testing.assert_array_equal(
                x2['state'][0][_ENCODING_LENGTH_BYTES:x2['state'][0, 0]+_ENCODING_LENGTH_BYTES],
                cv2.imencode('.png', np.array([0], dtype='uint8'))[1]
            )
            np.testing.assert_array_equal(
                x2['state'][1][_ENCODING_LENGTH_BYTES:x2['state'][1, 0]+_ENCODING_LENGTH_BYTES],
                cv2.imencode('.png', np.array([1], dtype='uint8'))[1]
            )
            self.assertEqual(len(x2['state'].shape), 2)
            self.assertEqual(x2['state'].shape[0], 2)
            self.assertEqual(x2['state'].shape[1], 74)
            self.assertEqual(set(x2.keys()), set(('state', 'something_else')))

        def test_encoding_exceeds_max_length(self):
            img_encoder = ImgEncoder('state', 68)
            x = {
                'state': np.array([[[0, 1, 2, 3]], [[0, 0, 0, 0]]], dtype='uint8'),
                'something_else': np.array([1,2,3])
            }
            x2 = img_encoder(x)
            np.testing.assert_array_equal(
                x2['state'][0][_ENCODING_LENGTH_BYTES:x2['state'][0, 0]+_ENCODING_LENGTH_BYTES],
                cv2.imencode('.png', np.array([[0, 0, 0, 0]], dtype='uint8'))[1]
            )
            self.assertEqual(len(x2['state'].shape), 2)
            self.assertEqual(x2['state'].shape[0], 1)
            self.assertEqual(x2['state'].shape[1], 72)
            self.assertEqual(set(x2.keys()), set(('state', 'something_else')))

    unittest.main()

