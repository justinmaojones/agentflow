import numpy as np
from agentflow.buffers.nd_array_buffer import NDArrayBuffer

class BufferMap(object):
    
    def __init__(self,max_length=1e6):
        self._max_length = int(max_length)
        self._n = 0
        self._buffer_cls = NDArrayBuffer
        self._buffers = {}
        self._index = 0

    def __len__(self):
        return self._n

    def append(self,data):

        if not (isinstance(data, dict) and len(data) > 0):
            raise TypeError("input to append must be a non-empty dictionary")

        if len(self._buffers) == 0:
            for k in data:
                self._buffers[k] = self._buffer_cls(self._max_length)
                self._buffers[k].append(data[k])
            shape = self.shape
            if not (len(set([v[1] for v in shape.values()])) == 1):
                raise TypeError("batch dim of all buffer elements must be the same")

            self.first_dim_size = list(shape.values())[0][0]

        else:

            for k in self._buffers:
                if k not in data:
                    raise TypeError("data must contain all elements of buffer")
                self._buffers[k].append(data[k])

        self._index = (self._index + 1) % self._max_length
        self._n = min(self._n + 1, self._max_length)

    def append_sequence(self,data):

        shape = {k: data[k].shape for k in data}
        assert len(set([(v[0],v[-1]) for v in shape.values()])) == 1, 'first and last dim of all data elements must be the same'
        shape_values = list(shape.values())[0]
        batch_size = shape_values[0]
        seq_size = shape_values[-1]

        if len(self._buffers) == 0:
            assert len(data) > 0
            for k in data:
                self._buffers[k] = self._buffer_cls(self._max_length)
                self._buffers[k].append_sequence(data[k])

            self.first_dim_size = batch_size

        else:
            assert len(data) == len(self._buffers), "data must contain all elements of buffer"
            for k in self._buffers:
                self._buffers[k].append_sequence(data[k])

        self._index = (self._index + seq_size) % self._max_length
        self._n = min(self._n + seq_size, self._max_length)

    def extend(self,X):
        for x in X:
            self.append(x)

    @property
    def shape(self):
        return {k:self._buffers[k].shape for k in self._buffers}

    def tail(self,seq_size,batch_idx=None):
        return {k:self._buffers[k].tail(seq_size,batch_idx) for k in self._buffers}

    def get(self,time_idx,batch_idx=None):
        return {k:self._buffers[k].get(time_idx,batch_idx) for k in self._buffers}

    def get_sequence(self,time_idx,seq_size,batch_idx=None):
        return {k:self._buffers[k].get_sequence(time_idx,seq_size,batch_idx) for k in self._buffers}

    def sample(self,nsamples):
        idx_time = np.random.choice(self._n-1,size=nsamples,replace=True)
        idx_batch = np.random.choice(self.first_dim_size,size=nsamples,replace=True)
        output = {k:self._buffers[k].get(idx_time,idx_batch) for k in self._buffers}
        return output


if __name__ == '__main__':
    import unittest

    class Test(unittest.TestCase):

        def test_creation(self):
            n = 10
            buf = BufferMap(n)
            buf.append({'x':np.random.randn(2),'y':np.random.randn(2,3),'z':np.random.randn(2,3,4)})
            self.assertEqual(buf.shape,{'x':(1,2),'y':(1,2,3),'z':(1,2,3,4)})

        def test_append(self):
            n = 5
            buf = BufferMap(n)
            for i in range(10):
                buf.append({'x':i*np.ones(2),'y':i*np.ones((2,3))})
                self.assertEqual(len(buf),min(i+1,n))
                self.assertEqual(buf._index,(i+1)%n)
            np.testing.assert_array_equal(buf._buffers['x']._buffer,np.arange(5,10)[:,None]*np.ones((1,2)))
            np.testing.assert_array_equal(buf._buffers['y']._buffer,np.arange(5,10)[:,None,None]*np.ones((1,2,3)))

            with self.assertRaises(TypeError):
                buf.append(1)

            with self.assertRaises(TypeError):
                buf.append({})

            with self.assertRaises(TypeError):
                buf.append({'x':np.ones(2)})

            with self.assertRaises(TypeError):
                BufferMap(n).append({'x':np.ones(2),'y':np.ones((3,3))})

        def test_shape(self):
            n = 5
            buf = BufferMap(n)
            for i in range(10):
                buf.append({'x':np.random.randn(2),'y':np.random.randn(2,3),'z':np.random.randn(2,3,4)})
                m = min(i+1,n)
                self.assertEqual(buf.shape,{'x':(m,2),'y':(m,2,3),'z':(m,2,3,4)})

        def test_tail(self):
            n = 5
            buf = BufferMap(n)
            for i in range(10):
                buf.append({'x':np.random.randn(2),'y':np.random.randn(2,3),'z':np.random.randn(2,3,4)})
                if i > 1:
                    tail = buf.tail(2)
                    self.assertEqual(tail['x'].shape, (2,2))
                    self.assertEqual(tail['y'].shape, (2,2,3))
                    self.assertEqual(tail['z'].shape, (2,2,3,4))

        def test_get(self):
            n = 5
            buf = BufferMap(n)
            for i in range(10):
                buf.append({'x':np.random.randn(2),'y':np.random.randn(2,3),'z':np.random.randn(2,3,4)})
                if i > 1:
                    output = buf.get(2)
                    self.assertEqual(output['x'].shape, (2,))
                    self.assertEqual(output['y'].shape, (2,3))
                    self.assertEqual(output['z'].shape, (2,3,4))

                    output = buf.get(2,1)
                    self.assertEqual(output['x'].shape, ())
                    self.assertEqual(output['y'].shape, (3,))
                    self.assertEqual(output['z'].shape, (3,4))

        def test_sample(self):
            n = 5
            buf = BufferMap(n)
            for i in range(10):
                buf.append({'x':np.random.randn(2),'y':np.random.randn(2,3),'z':np.random.randn(2,3,4)})
                if i > 1:
                    sample = buf.sample(5)
                    self.assertEqual(sample['x'].shape, (5,))
                    self.assertEqual(sample['y'].shape, (5,3))
                    self.assertEqual(sample['z'].shape, (5,3,4))

    unittest.main()

