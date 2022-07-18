import numpy as np
import cv2
import unittest

from agentflow.transform.img_compression import ImgEncoder
from agentflow.transform.img_compression import ImgDecoder

class TestEncoder(unittest.TestCase):

    def test_normal(self):
        img_encoder = ImgEncoder('state', 70)
        x = {
            'state':  np.array([[[0]], [[1]]], dtype='uint8'), 
            'something_else': np.array([1,2,3])
        }
        x2 = img_encoder(x)

        expected_0 = cv2.imencode('.png', np.array([0], dtype='uint8'))[1]
        expected_1 = cv2.imencode('.png', np.array([1], dtype='uint8'))[1]
        np.testing.assert_array_equal(
            x2['state'][0][:len(expected_0)],
            expected_0
        )
        np.testing.assert_array_equal(
            x2['state'][1][:len(expected_1)],
            expected_1
        )
        self.assertEqual(len(x2['state'].shape), 2)
        self.assertEqual(x2['state'].shape[0], 2)
        self.assertEqual(x2['state'].shape[1], 70)
        self.assertEqual(set(x2.keys()), set(('state', 'something_else')))

    def test_encoding_exceeds_max_length(self):
        img_encoder = ImgEncoder('state', 1)
        x = {
            'state': np.array([[[0, 1, 2, 3]], [[0, 0, 0, 0]]], dtype='uint8'),
            'something_else': np.array([1,2,3])
        }
        with self.assertRaises(ValueError):
            x2 = img_encoder(x)

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
