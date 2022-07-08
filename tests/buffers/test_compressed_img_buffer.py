import numpy as np
import unittest

from agentflow.buffers import BufferMap
from agentflow.buffers import CompressedImageBuffer


class TestCompressedImageBuffer(unittest.TestCase):

    def test_append_and_get(self):
        buffer = BufferMap(10)
        buffer = CompressedImageBuffer(buffer, keys_to_encode = ['state', 'state2'])

        x = {
            'state': np.arange(128).reshape((8, 4, 4)).astype('uint8'),
            'state2': np.arange(128, 256).reshape((8, 4, 4)).astype('uint8'),
            'something_else': np.arange(8)
        }
        # append twice
        buffer.append(x)
        buffer.append(x)

        # don't specify batch_idx, so output shape should be [2, 8, 4, 4]
        x_get = buffer.get(np.array([0,1]))
        self.assertEqual(set(x_get.keys()), set(('state', 'something_else', 'state2')))
        self.assertEqual(len(x_get['state']), 2)
        self.assertEqual(len(x_get['state2']), 2)
        np.testing.assert_array_equal(x_get['state'][0], x['state'])
        np.testing.assert_array_equal(x_get['state'][1], x['state'])
        np.testing.assert_array_equal(x_get['state2'][0], x['state2'])
        np.testing.assert_array_equal(x_get['state2'][1], x['state2'])

        # specify batch_idx, so output shape should be [2, 4, 4]
        x_get = buffer.get(np.array([0,0]), np.array([0,1]))
        self.assertEqual(set(x_get.keys()), set(('state', 'something_else', 'state2')))
        self.assertEqual(len(x_get['state']), 2)
        self.assertEqual(len(x_get['state2']), 2)
        np.testing.assert_array_equal(x_get['state'], x['state'][:2])
        np.testing.assert_array_equal(x_get['state2'], x['state2'][:2])

    def test_sample(self):
        buffer = BufferMap(10)
        buffer = CompressedImageBuffer(buffer, keys_to_encode = ['state', 'state2'])

        x = {
            'state':  np.array([[[0]], [[1]]], dtype='uint8'), 
            'state2':  np.array([[[2]], [[3]]], dtype='uint8'), 
            'something_else': np.array([1,2])
        }
        buffer.append(x)

        s = buffer.sample(2)

        self.assertEqual(set(s.keys()), set(('state', 'something_else', 'state2')))
        np.testing.assert_array_equal(s['state'].shape[1:], x['state'].shape[1:])
        np.testing.assert_array_equal(s['state2'].shape[1:], x['state2'].shape[1:])

    def test_tail(self):
        buffer = BufferMap(10)
        buffer = CompressedImageBuffer(buffer, keys_to_encode = ['state', 'state2'])

        x1 = {
            'state':  np.array([[[0]], [[1]]], dtype='uint8'), 
            'state2':  np.array([[[2]], [[3]]], dtype='uint8'), 
            'something_else': np.array([1,2])
        }
        buffer.append(x1)

        x2 = {
            'state':  np.array([[[0]], [[1]]], dtype='uint8'), 
            'state2':  np.array([[[2]], [[3]]], dtype='uint8'), 
            'something_else': np.array([1,2])
        }
        buffer.append(x2)

        # don't specify batch_idx, output shape should be [2, 2, 1, 1]
        x_tail = buffer.tail(2)
        self.assertEqual(set(x_tail.keys()), set(('state', 'something_else', 'state2')))
        np.testing.assert_array_equal(x_tail['state'][0], x1['state'])
        np.testing.assert_array_equal(x_tail['state'][1], x2['state'])
        np.testing.assert_array_equal(x_tail['state2'][0], x1['state2'])
        np.testing.assert_array_equal(x_tail['state2'][1], x2['state2'])

        # specify batch_idx, output shape should be [2, 2, 1, 1]
        x_tail = buffer.tail(2, np.array([0,1]))
        np.testing.assert_array_equal(x_tail['state'][0], x1['state'])
        np.testing.assert_array_equal(x_tail['state'][1], x2['state'])
        np.testing.assert_array_equal(x_tail['state2'][0], x1['state2'])
        np.testing.assert_array_equal(x_tail['state2'][1], x2['state2'])

if __name__ == '__main__':
    unittest.main()

