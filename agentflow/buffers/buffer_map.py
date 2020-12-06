import numpy as np
from .nd_array_buffer import NDArrayBuffer

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

        if len(self._buffers) == 0:
            assert len(data) > 0
            for k in data:
                self._buffers[k] = self._buffer_cls(self._max_length)
                self._buffers[k].append(data[k])
            shape = self.shape
            assert len(set([v[0] for v in shape.values()])) == 1, 'first dim of all buffer elements must be the same'
            self.first_dim_size = list(shape.values())[0][0]

        else:
            assert len(data) == len(self._buffers), "data must contain all elements of buffer"
            for k in self._buffers:
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


