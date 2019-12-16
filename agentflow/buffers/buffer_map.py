import numpy as np
from .nd_array_buffer_last_dim import NDArrayBufferLastDim
from .segment_tree import CircularSumTree, CircularMinTree

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

class PrioritizedBufferMap(BufferMap):

    def __init__(self,max_length=2**20,last_dim=True,alpha=1.,eps=1e-6,wclip=128.):
        super(PrioritizedBufferMap,self).__init__(max_length,last_dim)

        assert alpha >= 0
        self._alpha = alpha
        self._eps = eps
        self._wclip = wclip

        assert max_length & (max_length-1) == 0, "max_length must be a power of 2"
        self._sum_tree = None 
        self._isum_tree = None 
        self._min_tree = None 

    def _smooth_and_warp_priority(self,priority):
        p = (np.abs(priority)+self._eps)**self._alpha
        ip = np.minimum(1./p,self._wclip)
        p = 1./ip
        return p

    def append(self,data,priority):
        super(PrioritizedBufferMap,self).append(data)

        if not self._sum_tree:
            self._sum_tree = [CircularSumTree(self.max_length) for i in range(self.first_dim_size)]
            self._isum_tree = [CircularSumTree(self.max_length) for i in range(self.first_dim_size)]
            self._min_tree = [CircularMinTree(self.max_length) for i in range(self.first_dim_size)]

        P = self._smooth_and_warp_priority(priority)
        for i,p in enumerate(P):
            self._sum_tree[i].append(p)
            self._isum_tree[i].append(1./p)
            self._min_tree[i].append(p)

    def extend(self,X,priorities):
        for x,p in zip(X,priorities):
            self.append(x,p)

    def sample(self,nsamples,beta=1):
        # get priorities by agent dim
        pr_sums = np.array([tree.sum() for tree in self._sum_tree])
        pr_sum = pr_sums.sum()

        # first dim = number of agents
        # last dim = time
        idx_batch = np.random.choice(self.first_dim_size,size=nsamples,replace=True,p=pr_sums/pr_sum)
        idx_time = np.array([self._sum_tree[i].sample() for i in idx_batch])
        output = {k:self.buffers[k].buffer[idx_batch,...,idx_time] for k in self.buffers}

        # sampled inverse priorities
        #pr = np.array([self._sum_tree[i][j] for i,j in zip(idx_batch,idx_time)])
        ipr = np.array([self._isum_tree[i][j] for i,j in zip(idx_batch,idx_time)])

        # importance weights
        #p = pr/pr_sum
        N = len(self._sum_tree)*len(self._sum_tree[0])
        ipr_sum = sum([tree.sum() for tree in self._isum_tree])
        w = ipr / ipr_sum * N
        #wn = w/w.sum()*len(self._sum_tree)*len(self._sum_tree[0])
        #w = (1./len(self)/p)**beta

        # normalized importance weights
        #pr_min = min([tree.min() for tree in self._min_tree])
        #p_min = pr_min/pr_sum
        #max_w = (1./len(self)/p_min)**beta
        #w_normed = w/max_w

        assert 'importance_weight' not in output
        output['importance_weight'] = w

        self._idx_batch = idx_batch
        self._idx_time = idx_time

        return output

    def update_priorities(self,priority):
        P = self._smooth_and_warp_priority(priority)
        for i,j,p in zip(self._idx_batch,self._idx_time,P):
            self._sum_tree[i][j] = p
            self._isum_tree[i][j] = 1./p
            self._min_tree[i][j] = p
        self._idx_batch = None
        self._idx_time = None
