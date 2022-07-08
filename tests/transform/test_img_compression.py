import numpy as np
import cv2
import unittest

from agentflow.transform.img_compression import _ENCODING_LENGTH_BYTES
from agentflow.transform.img_compression import _encode_to_uint8
from agentflow.transform.img_compression import _decode_from_uint8
from agentflow.transform.img_compression import ImgEncoder
from agentflow.transform.img_compression import ImgDecoder

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



if __name__ == '__main__':
    unittest.main()
