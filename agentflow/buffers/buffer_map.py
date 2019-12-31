import numpy as np
from .nd_array_buffer_last_dim import NDArrayBufferLastDim


class BufferMap(object):
    
    def __init__(self,max_length=2**20,last_dim=True):
        self.max_length = max_length
        self._n = 0
        self.buffer_cls = NDArrayBufferLastDim 
        self.buffers = {}

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
            for k in self.buffers:
                self.buffers[k].append(data[k])
        self._n = min(self._n+1,self.max_length)

    def extend(self,X):
        for x in X:
            self.append(x)

    def shape(self):
        return {k:self.buffers[k].shape() for k in self.buffers}

    def sample(self,nsamples):
        # first dim = number of agents
        # last dim = time
        idx_batch = np.random.choice(self.first_dim_size,size=nsamples,replace=True)
        idx_time = np.random.choice(self._n-1,size=nsamples,replace=True)
        output = {k:self.buffers[k].buffer[idx_batch,...,idx_time] for k in self.buffers}
        return output

