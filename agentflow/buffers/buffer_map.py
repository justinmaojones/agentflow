import numpy as np
from .nd_array_buffer_last_dim import NDArrayBufferLastDim
from ..common.prefix_sum_tree import PrefixSumTree


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

class CircularSumTree(PrefixSumTree):

    def __new__(self,agent_dim_size,time_dim_size):
        return PrefixSumTree((agent_dim_size,time_dim_size)).view(CircularSumTree)


    def __init__(self,agent_dim_size,time_dim_size):
        self._index = 0
        self.agent_dim_size = agent_dim_size
        self.time_dim_size = time_dim_size
        self._full = False
        
    def append(self,val):
        if self._full:
            dropped = self[:,self._index]
        else:
            dropped = None
        self[:,self._index] = val
        self._index = (self._index + 1) % self.shape[1]
        if self._index == 0:
            self._full = True
        return dropped

class PrioritizedBufferMap(BufferMap):

    def __init__(self,max_length=2**20,last_dim=True,alpha=1.,eps=1e-6,wclip=128.):
        super(PrioritizedBufferMap,self).__init__(max_length,last_dim)

        assert alpha >= 0
        self._alpha = alpha
        self._eps = eps
        self._wclip = wclip

        assert max_length & (max_length-1) == 0, "max_length must be a power of 2"
        self._sum_tree = None 
        self._inv_sum = 0.

    def _smooth_and_warp_priority(self,priority):
        p = (np.abs(priority)+self._eps)**self._alpha
        ip = np.minimum(1./p,self._wclip)
        p = 1./ip
        return p

    def append(self,data,priority):
        super(PrioritizedBufferMap,self).append(data)

        if self._sum_tree is None:
            self._sum_tree = CircularSumTree(self.first_dim_size,self.max_length)

        p = self._smooth_and_warp_priority(priority)
        self._inv_sum += (1./p).sum()
        dropped = self._sum_tree.append(p)
        if dropped is not None:
            self._inv_sum -= (1./dropped).sum()

    def extend(self,X,priorities):
        for x,p in zip(X,priorities):
            self.append(x,p)

    def sample(self,nsamples,beta=1):
        # get priorities by agent dim
        pr_sums = np.array([tree.sum() for tree in self._sum_tree])
        pr_sum = pr_sums.sum()

        # first dim = number of agents
        # last dim = time
        idx_sample = self._sum_tree.sample(nsamples)
        idx_batch = idx_sample // self._sum_tree.time_dim_size
        idx_time = idx_sample - (idx_batch * self._sum_tree.time_dim_size)
        output = {k:self.buffers[k].buffer[idx_batch,...,idx_time] for k in self.buffers}

        # sampled inverse priorities
        ipr = 1./self._sum_tree[idx_batch,idx_time]

        # importance weights
        N = self._sum_tree.size
        ipr_sum = self._inv_sum
        w = (ipr / ipr_sum * N) ** beta

        assert 'importance_weight' not in output
        output['importance_weight'] = w

        self._idx_batch = idx_batch
        self._idx_time = idx_time

        return output

    def update_priorities(self,priority):
        p = self._smooth_and_warp_priority(priority)
        dropped = self._sum_tree[self._idx_batch,self._idx_time]
        self._sum_tree[self._idx_batch,self._idx_time] = p
        self._inv_sum += (1./p).sum() - (1./dropped).sum()
        self._idx_batch = None
        self._idx_time = None
