import numpy as np
import cv2

from agentflow.transform.img_encoder import _ENCODING_LENGTH_BYTES


def _decode_length(x):
    y = 0
    for e in x:
        y |= e
    return y

class ImgDecoder(object):

    def __init__(self, key_to_decode):
        self.key_to_decode = key_to_decode

    def transform(self, data):
        x_encoded = data[self.key_to_decode]
        assert x_encoded.dtype == np.uint8, "encoded data must be uint8"
        assert x_encoded.ndim == 2, "encoded data must be 1d, excluding batch dimension"
        n = len(x_encoded)
        decodings = []
        for i in range(n):
            # length of encoding
            xl = _decode_length(x_encoded[i, :_ENCODING_LENGTH_BYTES])
            # decode
            left = _ENCODING_LENGTH_BYTES
            right = xl + _ENCODING_LENGTH_BYTES
            decodings.append(cv2.imdecode(x_encoded[i][left:right], cv2.IMREAD_UNCHANGED))
        decoded_array = np.stack(decodings)
        output = {k: decoded_array if k==self.key_to_decode else data[k] for k in data}
        return output

    def __call__(self, data):
        return self.transform(data)

if __name__ == '__main__':
    from agentflow.transform.img_encoder import ImgEncoder
    import unittest

    class Test(unittest.TestCase):

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

