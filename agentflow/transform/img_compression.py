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



if __name__ == '__main__':
    import unittest

    class TestEncodeDecodeUInt8(unittest.TestCase):

        def test_encode_to_uint8(self):

            np.testing.assert_array_equal(_encode_to_uint8(0), np.array([0, 0, 0, 0], dtype='uint8'))
            np.testing.assert_array_equal(_encode_to_uint8(593), np.array([81, 2, 0, 0], dtype='uint8'))
            np.testing.assert_array_equal(_encode_to_uint8(2**32-1), np.array([255, 255, 255, 255], dtype='uint8'))

            np.testing.assert_array_equal(
                _encode_to_uint8(np.array([0, 593, 2**32-1])),
                np.array(
                    [[0, 0, 0, 0], 
                     [81, 2, 0, 0], 
                     [255, 255, 255, 255]],
                    dtype='uint8')
            )
            
        def test_decode_to_uint8(self):

            self.assertEqual(_decode_from_uint8(np.array([0, 0, 0, 0], dtype='uint8')), int(0))
            self.assertEqual(_decode_from_uint8(np.array([81, 2, 0, 0], dtype='uint8')), int(593))
            self.assertEqual(_decode_from_uint8(np.array([255, 255, 255, 255], dtype='uint8')), int(2**32-1))

            np.testing.assert_array_equal(
                np.array([0, 593, 2**32-1]),
                _decode_from_uint8(
                    np.array(
                        [[0, 0, 0, 0], 
                         [81, 2, 0, 0], 
                         [255, 255, 255, 255]],
                        dtype='uint8')
                )
            )

            x = np.array([0, 593, 2**32-1], dtype='int64'),
            np.testing.assert_array_equal(_decode_from_uint8(_encode_to_uint8(x)), x)
            
    class TestEncoder(unittest.TestCase):

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

    class TestDecoder(unittest.TestCase):

        def test_normal(self):
            img_encoder = ImgEncoder('state', 3000)
            img_decoder = ImgDecoder('state')
            x = {
                'state':  np.random.choice(10,size=(3,4,5)).astype('uint8'),
                'something_else': np.array([1,2,3])
            }
            x2 = img_decoder(img_encoder(x))
            np.testing.assert_array_equal(x2['state'], x['state'])
            np.testing.assert_array_equal(x2['something_else'], x['something_else'])
            self.assertEqual(set(x2.keys()), set(('state', 'something_else')))



    unittest.main()

