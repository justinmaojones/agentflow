from __future__ import annotations

import numpy as np
from typing import Dict, Union

from agentflow.buffers.flow import BufferFlow
from agentflow.buffers.source import BufferSource
from agentflow.transform import ImgDecoder
from agentflow.transform import ImgEncoder


class CompressedImageBuffer(BufferFlow):

    def __init__(self,
            source: Union[BufferFlow, BufferSource],
            max_encoding_size: int = 2000,
            keys_to_encode: list[str] = ['state', 'state2'],
        ):
        
        self.source = source
        self._max_encoding_size = max_encoding_size
        self._keys_to_encode = keys_to_encode

        self._encoders = [ImgEncoder(k, max_encoding_size) for k in keys_to_encode]
        self._decoders = [ImgDecoder(k) for k in keys_to_encode]

    def encode(self, data: Dict[str, np.ndarray]):
        for encoder in self._encoders:
            data = encoder(data)
        return data

    def decode(self, data: Dict[str, np.ndarray]):
        for decoder in self._decoders:
            data = decoder(data)
        return data

    def _flattened_decode(self, data: Dict[str, np.ndarray]):
        # flatten time and batch dim
        s = {}
        for k in self._keys_to_encode:
            s[k] = data[k].shape
            data[k] = data[k].reshape([s[k][0]*s[k][1], *s[k][2:]])

        # decode
        data = self.decode(data)

        # unflatten
        for k in self._keys_to_encode:
            data[k] = data[k].reshape(s[k][:2] + data[k].shape[1:])

        return data

    def append(self, data: Dict[str, np.ndarray]):
        self.source.append(self.encode(data))

    def append_sequence(self, data: Dict[str, np.ndarray]):
        raise NotImplementedError

    def get(self, time_idx: np.ndarray, batch_idx: np.ndarray = None):
        if batch_idx is None:
            return self._flattened_decode(self.source.get(time_idx))
        else:
            return self.decode(self.source.get(time_idx, batch_idx))

    def get_sequence(self, time_idx: np.ndarray, seq_size: int, batch_idx: np.ndarray = None):
        raise NotImplementedError

    def sample(self, n_samples: int):
        return self.decode(self.source.sample(n_samples))

    def tail(self, seq_size: int, batch_idx: np.ndarray = None):
        return self._flattened_decode(self.source.tail(seq_size, batch_idx))


if __name__ == '__main__':
    import unittest
    from agentflow.buffers import BufferMap

    class Test(unittest.TestCase):

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


    unittest.main()

