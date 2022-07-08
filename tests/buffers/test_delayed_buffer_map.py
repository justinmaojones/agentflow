import numpy as np
import unittest

from agentflow.buffers import BufferMap 
from agentflow.buffers import DelayedBufferMap

class TestDelayedBufferMap(unittest.TestCase):

    def test_all(self):
        n = 10
        source = BufferMap(n)
        buf = DelayedBufferMap(source)

        # append data, but nothing ready to publish yet
        x = {
            'x': np.array([1, 2]),
            'done': np.array([0, 0]),
        }
        buf.append(x)
        self.assertEqual(len(buf.source), 0)
        self.assertTrue(np.all(buf._delayed_buffer_map._count_since_last_publish==np.array([1, 1])))

        # append data, element 1 ready to publish 
        x = {
            'x': np.array([3, 4]),
            'done': np.array([1, 0]),
        }
        buf.append(x)
        self.assertTrue(np.all(buf.source['x']._buffer[:2]==np.array([[1, 3]]).T))
        self.assertTrue(np.all(buf.source['done']._buffer[:2]==np.array([[0, 1]]).T))
        self.assertTrue(np.all(buf._delayed_buffer_map._count_since_last_publish==np.array([0, 2])))

        # append data, both elements ready to publish 
        x = {
            'x': np.array([5, 6]),
            'done': np.array([1, 1]),
        }
        buf.append(x)
        self.assertTrue(np.all(buf.source['x']._buffer==np.array([[1, 3, 5, 2, 4, 6, 0, 0, 0, 0]]).T))
        self.assertTrue(np.all(buf.source['done']._buffer==np.array([[0, 1, 1, 0, 0, 1, 0, 0, 0, 0]]).T))
        self.assertTrue(np.all(buf._delayed_buffer_map._count_since_last_publish==np.array([0, 0])))

        # append data, wrap around 
        for i in range(0, 9):
            buf.append({'x': np.array([i, i+1]), 'done': np.array([0, 0])})

        self.assertTrue(np.all(buf.source['x']._buffer==np.array([[1, 3, 5, 2, 4, 6, 0, 0, 0, 0]]).T))
        self.assertTrue(np.all(buf.source['done']._buffer==np.array([[0, 1, 1, 0, 0, 1, 0, 0, 0, 0]]).T))
        self.assertTrue(np.all(buf._delayed_buffer_map._count_since_last_publish==np.array([9, 9])))

        x = {
            'x': np.array([9, 10]),
            'done': np.array([1, 0]),
        }
        buf.append(x)
        self.assertTrue(np.all(buf.source['x']._buffer==np.array([[4, 5, 6, 7, 8, 9, 0, 1, 2, 3]]).T))
        self.assertTrue(np.all(buf._delayed_buffer_map._count_since_last_publish==np.array([0, 10])))

if __name__ == '__main__':
    unittest.main()
