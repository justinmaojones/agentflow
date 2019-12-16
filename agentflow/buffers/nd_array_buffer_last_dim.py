import numpy as np
import random

class NDArrayBufferLastDim(object):
    
    def __init__(self,max_length=1e6):
        self.buffer = None
        self._index = 0
        self._n = 0
        self.max_length = int(max_length)
        
    def __len__(self):
        return self._n
    
    def append(self,x):
        if self.buffer is None:
            shape = tuple(list(x.shape) + [self.max_length])
            self.buffer = np.zeros(shape)
        ndim = self.buffer.ndim
        self.buffer[...,self._index] = x
        self._n = min(self._n+1,self.max_length)
        self._index = (self._index+1) % self.max_length

    def extend(self,X):
        for x in X:
            self.append(x)

    def get_sequence_slices(self,i,seq_size):
        assert i >= (seq_size-1)
        assert i < self._n
        j2 = self._index + i + 1
        j1 = j2 - seq_size
        n = self._n
        slices = []
        if j1 >= n:
            j1 = j1 % n
            j2 = j2 % n
            return [slice(j1,j2)]
        if j2 > n:
            j2 = j2 % n
            return [slice(j1,n),slice(0,j2)]
        return [slice(j1,j2)]

    def get(idx):
        idx = np.array(idx)
        if self._n == self.max_length:
            idx += self._i
        return self.buffer[...,idx % self._n]

    def get_sequence(self,i,seq_size):
        slices = self.get_sequence_slices(i,seq_size)
        ndim = self.buffer.ndim
        return np.concatenate([self.buffer[...,s] for s in slices],axis=-1)

    def sample(self,n=1,seq_size=1):
        assert n > 0
        assert seq_size > 0
        assert len(self) > 0
        assert len(self) >= seq_size

        output = []
        for i in range(n):
            index = random.randint(seq_size,len(self)-1)
            output.append(self.get_sequence(index,seq_size))
        return np.concatenate(output,axis=0)

    def shape(self):
        return self.buffer.shape


if __name__ == '__main__':
    import unittest

    class Test(unittest.TestCase):

        def test_all(self):
            n = 10
            buf = NDArrayBufferLastDim(n)
            self.assertEqual(len(buf), 0)
            for i in range(2*n-2):
                buf.append(np.random.randn(1,2,3))
                self.assertEqual(len(buf), min(n,i+1))
                self.assertEqual(buf._index, (i+1) % n)

            self.assertEqual(buf.buffer.shape, (1,2,3,n))

            self.assertEqual(buf.get_sequence_slices(0,1), [slice(buf._index,buf._index+1)])
            self.assertEqual(buf.get_sequence_slices(1,2), [slice(buf._index,buf._index+2)])
            self.assertEqual(buf.get_sequence_slices(1,2), [slice(buf._index,buf._index+2)])
            self.assertEqual(buf.get_sequence_slices(2,1), [slice(0,1)])
            self.assertEqual(buf.get_sequence_slices(2,2), [slice(buf._index+1,buf._index+2),slice(0,1)])
            self.assertEqual(buf.get_sequence_slices(3,4), [slice(buf._index,buf._index+2),slice(0,2)])
            self.assertEqual(buf.get_sequence_slices(3,2), [slice(0,2)])
            self.assertEqual(buf.get_sequence_slices(4,2), [slice(1,3)])

            self.assertTrue(
                np.all(
                    buf.get_sequence(3,4)==np.concatenate([
                            buf.buffer[:,:,:,buf._index:buf._index+2],
                            buf.buffer[:,:,:,:2]
                        ],axis=-1)))

            self.assertEqual(buf.sample(1,1).shape, (1,2,3,1))
            self.assertEqual(buf.sample(1,2).shape, (1,2,3,2))
            self.assertEqual(buf.sample(3,1).shape, (3,2,3,1))
            self.assertEqual(buf.sample(3,4).shape, (3,2,3,4))

    unittest.main()
