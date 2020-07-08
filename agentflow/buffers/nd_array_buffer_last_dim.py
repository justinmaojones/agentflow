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
            self.buffer = np.zeros(shape,dtype=x.dtype)
        self.buffer[...,self._index] = x
        self._n = min(self._n+1,self.max_length)
        self._index = (self._index+1) % self.max_length

    def append_sequence(self,x):
        if self.buffer is None:
            shape = tuple(list(x.shape)[:-1] + [self.max_length])
            self.buffer = np.zeros(shape,dtype=x.dtype)
        seq_size = x.shape[-1] 
        i1 = self._index
        i2 = min(self._index + seq_size, self.max_length)
        segment_size = i2 - i1 
        self.buffer[...,i1:i2] = x[...,:segment_size]
        self._n = min(self._n + segment_size, self.max_length)
        self._index = (self._index + segment_size) % self.max_length
        if segment_size < seq_size:
            self.append_sequence(x[...,segment_size:])

    def extend(self,X):
        for x in X:
            self.append(x)

    def _tail_slices(self,seq_size):
        assert seq_size <= self._n
        slices = []
        i1 = max(0, self._index - seq_size)
        i2 = self._index
        slices.append(slice(i1,i2))
        seq_size -= (i2-i1)
        if seq_size > 0:
            i1 = self._n - seq_size
            i2 = self._n
            slices.append(slice(i1,i2))
            assert i2-i1 == seq_size
        return list(reversed(slices))

    def tail(self,seq_size,batch_idx=None):
        slices = self._tail_slices(seq_size)
        if batch_idx is None:
            return np.concatenate([self.buffer[...,s] for s in slices],axis=-1)
        else:
            return np.concatenate([self.buffer[batch_idx,...,s] for s in slices],axis=-1)

    def get_sequence_slices(self,t,seq_size):
        # todo: this is wrong...doesn't work when t=self._index-1
        # t is relative to self._index
        assert False, "todo: this is wrong...doesn't work when t=self._index-1"
        assert t >= (seq_size-1)
        assert t < self._n
        j2 = self._index + t + 1
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

    def get(self,idx):
        idx = np.array(idx)
        if self._n == self.max_length:
            idx += self._index
        return self.buffer[...,idx % self._n]

    def get_sequence(self,time_idx,seq_size,batch_idx=None):
        slices = self.get_sequence_slices(time_idx,seq_size)
        if batch_idx is None:
            return np.concatenate([self.buffer[...,s] for s in slices],axis=-1)
        else:
            return np.concatenate([self.buffer[batch_idx,...,s] for s in slices],axis=-1)

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

        def test_tail_slices(self):
            n = 10
            buf = NDArrayBufferLastDim(n)
            for i in range(3):
                buf.append(np.array([i]))

            self.assertEqual(buf._tail_slices(2), [slice(1,3)])
            self.assertEqual(buf._tail_slices(3), [slice(0,3)])

            for i in range(3,13):
                buf.append(np.array([i]))

            self.assertEqual(buf._index, 3) 
            self.assertEqual(buf._tail_slices(2), [slice(1,3)])
            self.assertEqual(buf._tail_slices(3), [slice(0,3)])
            self.assertEqual(buf._tail_slices(4), [slice(9,10),slice(0,3)])
            self.assertEqual(buf._tail_slices(10), [slice(3,10),slice(0,3)])

        def test_tail(self):
            n = 3 
            buf = NDArrayBufferLastDim(n)
            buf.append(np.array([1,2]))
            buf.append(np.array([3,4]))

            # test tail prior to filling up buffer
            self.assertTrue(np.all(buf.tail(1) == np.array([[3],[4]])))
            self.assertTrue(np.all(buf.tail(2) == np.array([[1,3],[2,4]])))

            buf.append(np.array([5,6]))

            # test tail when buffer is full 
            self.assertTrue(np.all(buf.tail(2) == np.array([[3,5],[4,6]])))

            buf.append(np.array([7,8]))

            # test tail when buffer is full and has been overwritten 
            self.assertTrue(np.all(buf.tail(1) == np.array([[7],[8]])))
            self.assertTrue(np.all(buf.tail(2) == np.array([[5,7],[6,8]])))
            self.assertTrue(np.all(buf.tail(3) == np.array([[3,5,7],[4,6,8]])))

            # test batch_idx
            self.assertTrue(np.all(buf.tail(3,0) == np.array([[3,5,7]])))
            self.assertTrue(np.all(buf.tail(3,[0]) == np.array([[3,5,7]])))
            self.assertTrue(np.all(buf.tail(3,[1,0]) == np.array([[4,6,8],[3,5,7]])))

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

        def test_append_sequence(self):
            n = 10
            buf = NDArrayBufferLastDim(n)

            x = np.arange(3)[None]
            buf.append_sequence(x)
            self.assertTrue(np.all(buf.buffer[...,:3] == x))
            self.assertEqual(buf._index, 3)
            self.assertEqual(buf._n, 3)

            x = np.arange(3,3+10)[None]
            buf.append_sequence(x)
            self.assertTrue(np.all(buf.buffer == np.concatenate([x[...,7:],x[...,:7]],axis=-1)))
            self.assertEqual(buf._index, 3)
            self.assertEqual(buf._n, 10)

            x = np.arange(13,13+13)[None]
            buf.append_sequence(x)
            self.assertTrue(np.all(buf.buffer == np.concatenate([x[...,13-6:],x[...,3:13-6]],axis=-1)))
            self.assertEqual(buf._index, 6)
            self.assertEqual(buf._n, 10)

    unittest.main()
