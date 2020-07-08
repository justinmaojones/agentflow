import numpy as np
from .nd_array_buffer_last_dim import NDArrayBufferLastDim


class BufferMap(object):
    
    def __init__(self,max_length=2**20):
        self.max_length = max_length
        self._n = 0
        self.buffer_cls = NDArrayBufferLastDim 
        self.buffers = {}
        self._index = 0

    def __len__(self):
        return self._n

    def append(self,data):

        if len(self.buffers) == 0:
            assert len(data) > 0
            for k in data:
                self.buffers[k] = self.buffer_cls(self.max_length)
                self.buffers[k].append(data[k])
            shape = self.shape()
            assert len(set([v[0] for v in shape.values()])) == 1, 'first dim of all buffer elements must be the same'
            self.first_dim_size = list(shape.values())[0][0]

        else:
            assert len(data) == len(self.buffers), "data must contain all elements of buffer"
            for k in self.buffers:
                self.buffers[k].append(data[k])

        self._index = (self._index + 1) % self.max_length
        self._n = min(self._n + 1, self.max_length)

    def append_sequence(self,data):

        shape = {k: data[k].shape for k in data}
        assert len(set([(v[0],v[-1]) for v in shape.values()])) == 1, 'first and last dim of all data elements must be the same'
        shape_values = list(shape.values())[0]
        batch_size = shape_values[0]
        seq_size = shape_values[-1]

        if len(self.buffers) == 0:
            assert len(data) > 0
            for k in data:
                self.buffers[k] = self.buffer_cls(self.max_length)
                self.buffers[k].append_sequence(data[k])

            self.first_dim_size = batch_size

        else:
            assert len(data) == len(self.buffers), "data must contain all elements of buffer"
            for k in self.buffers:
                self.buffers[k].append_sequence(data[k])

        self._index = (self._index + seq_size) % self.max_length
        self._n = min(self._n + seq_size, self.max_length)

    def extend(self,X):
        for x in X:
            self.append(x)

    def shape(self):
        return {k:self.buffers[k].shape() for k in self.buffers}

    def tail(self,seq_size,batch_idx=None):
        return {k:self.buffers[k].tail(seq_size,batch_idx) for k in self.buffers}

    def get(self,time_idx):
        return {k:self.buffers[k].get(time_idx) for k in self.buffers}

    def get_sequence(self,time_idx,seq_size,batch_idx=None):
        return {k:self.buffers[k].get_sequence(time_idx,seq_size,batch_idx) for k in self.buffers}

    def sample(self,nsamples):
        # first dim = number of agents
        # last dim = time
        idx_batch = np.random.choice(self.first_dim_size,size=nsamples,replace=True)
        idx_time = np.random.choice(self._n-1,size=nsamples,replace=True)
        output = {k:self.buffers[k].buffer[idx_batch,...,idx_time] for k in self.buffers}
        return output

    def sample_backwards(self,nsamples,ntimesteps=None):
        if ntimesteps is None:
            ntimesteps = nsamples
        idx_batch = np.random.choice(self.first_dim_size,size=nsamples,replace=True)
        idx_time = np.random.randint(
            low = self._n + self._index - ntimesteps,
            high = self._n + self._index,
            size = nsamples
        ) % self._n
        output = {k:self.buffers[k].buffer[idx_batch,...,idx_time] for k in self.buffers}
        self._index = (self._n + self._index - ntimesteps) % self._n
        return output

    def reset_index(self):
        self._index = 0


