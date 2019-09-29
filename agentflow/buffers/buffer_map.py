import numpy as np
from .nd_array_buffer import NDArrayBuffer 

class BufferMap(object):
    
    def __init__(self,max_length=1e6):
        self.max_length = max_length
        self._n = 0
        self.buffers = {
            'reward':NDArrayBuffer(self.max_length),
            'action':NDArrayBuffer(self.max_length),
            'state':NDArrayBuffer(self.max_length),
            'done':NDArrayBuffer(self.max_length),
        }

    def __len__(self):
        return self._n

    def append(self,data):
        for k in self.buffers:
            assert k in data
            self.buffers[k].append(data[k])
        self._n += 1

    def extend(self,X):
        for x in X:
            self.append(x)

    def sample(self,nsamples):
        assert nsamples <= self._n-1 #because there should always exist a state2
        idx = np.random.choice(self._n-1,size=nsamples,replace=False)
        output = {k:self.buffers[k].get(idx) for k in self.buffers}
        output['state2'] = self.buffers['state'].get(idx+1)
        return output

        

