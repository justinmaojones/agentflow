import numpy as np
import cv2


class ImgDecoder(object):

    def __init__(self, key_to_decode):
        self.key_to_decode = key_to_decode

    @property
    def _key_encoded(self):
        return self.key_to_decode + '_encoded'

    @property
    def _key_encoding_length(self):
        return self.key_to_decode + '_encoding_length'

    def transform(self, data):
        x_encoded = data[self._key_encoded]
        x_enc_len = data[self._key_encoding_length]
        assert x_encoded.dtype == np.uint8, "encoded data must be uint8"
        assert x_enc_len.dtype in (np.int32, np.int64), "encoding length must be int32 or int64"
        assert x_encoded.ndim == 2, "encoded data must be 1d, excluding batch dimension"
        assert x_enc_len.ndim == 1, "encoded data lengths must be 1d, including batch dimension"
        assert len(x_encoded) == len(x_enc_len), "encoded data and lengths must have same batch size"
        n = len(x_encoded)
        decodings = []
        for i in range(n):
            xl = x_enc_len[i]
            decodings.append(cv2.imdecode(x_encoded[i][:xl], cv2.IMREAD_UNCHANGED))
        output = {k:data[k] for k in data}
        output.pop(self._key_encoded)
        output.pop(self._key_encoding_length)
        assert self.key_to_decode not in output
        output[self.key_to_decode] = np.stack(decodings)
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
            x = {'state':  np.random.choice(10,size=(3,4,5)).astype('uint8')}
            x2 = img_decoder(img_encoder(x))
            np.testing.assert_array_equal(x2['state'], x['state'])
            self.assertEqual(set(x2.keys()), set(('state',)))

    unittest.main()

