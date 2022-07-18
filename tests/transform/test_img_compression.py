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

        # check encodings
        np.testing.assert_array_equal(
            x2['state'][0][:len(expected_0)],
            expected_0
        )
        np.testing.assert_array_equal(
            x2['state'][1][:len(expected_1)],
            expected_1
        )

        # check padding
        np.testing.assert_array_equal(
            x2['state'][0][len(expected_0):],
            np.zeros(70-len(expected_0), dtype='uint8')
        )
        np.testing.assert_array_equal(
            x2['state'][1][len(expected_1):],
            np.zeros(70-len(expected_1), dtype='uint8')
        )

        # check shapes
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

    def test_reshape(self):
        img_encoder = ImgEncoder('state', 1000, reshape=(4, 5*6))
        x = {
            'state': np.random.choice(256, size=(3,4,5,6)).astype('uint8'),
            'something_else': np.array([1,2,3])
        }
        x2 = img_encoder(x)
        
        for i in range(3):
            expected = cv2.imencode('.png', np.array(x['state'][i].reshape(4, 30), dtype='uint8'))[1]
            np.testing.assert_array_equal(
                x2['state'][i][:len(expected)],
                expected
            )
            np.testing.assert_array_equal(
                x2['state'][i][len(expected):],
                np.zeros(1000-len(expected), dtype='uint8')
            )

        self.assertEqual(len(x2['state'].shape), 2)
        self.assertEqual(x2['state'].shape[0], 3)
        self.assertEqual(x2['state'].shape[1], 1000)
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

    def test_reshape(self):
        img_encoder = ImgEncoder('state', 3000, reshape=(4, 5*6))
        img_decoder = ImgDecoder('state', reshape=(4, 5, 6))
        x = {
            'state': np.random.choice(256, size=(3,4,5,6)).astype('uint8'),
            'something_else': np.array([1,2,3])
        }
        x2 = img_decoder(img_encoder(x))

        np.testing.assert_array_equal(x2['state'], x['state'])
        np.testing.assert_array_equal(x2['something_else'], x['something_else'])
        self.assertEqual(set(x2.keys()), set(('state', 'something_else')))



if __name__ == '__main__':
    unittest.main()
