import numpy as np
import cv2

class ImgEncoder(object):

    def __init__(self, key_to_encode, max_encoding_size):
        self.max_encoding_size = max_encoding_size
        self.key_to_encode = key_to_encode

    @property
    def _key_encoded(self):
        return self.key_to_encode + '_encoded'

    @property
    def _key_encoding_length(self):
        return self.key_to_encode + '_encoding_length'

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
        encoded = np.zeros((m, self.max_encoding_size), dtype=np.uint8)
        lengths = np.zeros(m, dtype=int)
        for i in range(m):
            e = encodings[i]
            encoded[i, :len(e)] = e
            lengths[i] = len(e)

        output = {k:data[k] for k in data}
        output.pop(self.key_to_encode)
        assert self._key_encoded not in data
        assert self._key_encoding_length not in data
        output[self._key_encoded] = encoded
        output[self._key_encoding_length] = lengths
        return output

    def __call__(self, data):
        return self.transform(data)

if __name__ == '__main__':
    import unittest

    class Test(unittest.TestCase):

        def test_normal(self):
            img_encoder = ImgEncoder('state', 70)
            x = {'state':  np.array([[[0]], [[1]]], dtype='uint8')}
            x2 = img_encoder(x)
            np.testing.assert_array_equal(
                x2['state_encoded'][0][:x2['state_encoding_length'][0]],
                cv2.imencode('.png', np.array([0], dtype='uint8'))[1]
            )
            np.testing.assert_array_equal(
                x2['state_encoded'][1][:x2['state_encoding_length'][1]],
                cv2.imencode('.png', np.array([1], dtype='uint8'))[1]
            )
            self.assertEqual(len(x2['state_encoded']), 2)
            self.assertEqual(len(x2['state_encoding_length']), 2)
            self.assertEqual(set(x2.keys()), set(('state_encoded', 'state_encoding_length')))

        def test_encoding_exceeds_max_length(self):
            img_encoder = ImgEncoder('state', 68)
            x = {'state': np.array([[[0, 1, 2, 3]], [[0, 0, 0, 0]]], dtype='uint8')}
            x2 = img_encoder(x)
            np.testing.assert_array_equal(
                x2['state_encoded'][0][:x2['state_encoding_length'][0]],
                cv2.imencode('.png', np.array([[0, 0, 0, 0]], dtype='uint8'))[1]
            )
            self.assertEqual(len(x2['state_encoded']), 1)
            self.assertEqual(len(x2['state_encoding_length']), 1)
            self.assertEqual(set(x2.keys()), set(('state_encoded', 'state_encoding_length')))

    unittest.main()

