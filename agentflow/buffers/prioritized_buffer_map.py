import numpy as np
from .prefix_sum_tree import PrefixSumTree
from .buffer_map import BufferMap

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

    def append_sequence(self,x):
        seq_size = x.shape[-1] 
        i1 = self._index
        i2 = min(self._index + seq_size, self.time_dim_size)
        segment_size = i2 - i1 
        self[...,i1:i2] = x[...,:segment_size]
        if not self._full and self._index + segment_size >= self.time_dim_size:
            self._full = True
        self._index = (self._index + segment_size) % self.time_dim_size 
        if segment_size < seq_size:
            self.append_sequence(x[...,segment_size:])


class PrioritizedBufferMap(BufferMap):

    def __init__(self,max_length=2**20,alpha=0.6,eps=1e-4,wclip=32.,n_beta_annealing_steps=None):
        super(PrioritizedBufferMap,self).__init__(max_length)

        assert alpha >= 0
        self._alpha = alpha
        self._eps = eps
        self._wclip = wclip
        self._n_beta_annealing_steps = n_beta_annealing_steps
        self._counter_for_beta = 0

        self._sum_tree = None 
        self._inv_sum = 0.

    def _smooth_and_warp_priority(self,priority):
        p = (np.abs(priority)+self._eps)**self._alpha
        #ip = np.minimum(1./p,self._wclip)
        #p = 1./ip
        return p

    def append(self,data,priority=None):
        super(PrioritizedBufferMap,self).append(data)

        if priority is None:
            priority = np.ones((self.first_dim_size,1))

        if self._sum_tree is None:
            self._sum_tree = CircularSumTree(self.first_dim_size,self.max_length)

        p = self._smooth_and_warp_priority(priority)
        self._inv_sum += (1./p).sum()
        dropped = self._sum_tree.append(p)
        if dropped is not None:
            self._inv_sum -= (1./dropped).sum()

    def append_sequence(self,data,priority=None):
        super(PrioritizedBufferMap,self).append_sequence(data)

        if priority is None:
            t = list(data.values())[0].shape[-1]
            priority = np.ones((self.first_dim_size,t))

        if self._sum_tree is None:
            self._sum_tree = CircularSumTree(self.first_dim_size,self.max_length)

        p = self._smooth_and_warp_priority(priority)
        self._inv_sum += (1./p).sum()
        dropped = self._sum_tree.append_sequence(p)
        if dropped is not None:
            self._inv_sum -= (1./dropped).sum()

        assert self._index == self._sum_tree._index, "index mismatch (%d, %d)" % (self._index, self._sum_tree._index)

    def extend(self,X,priorities):
        for x,p in zip(X,priorities):
            self.append(x,p)

    def sample(self,nsamples,beta=None):

        # first dim = number of agents
        # last dim = time
        idx_sample = self._sum_tree.sample(nsamples)
        idx_batch = idx_sample // self._sum_tree.time_dim_size
        idx_time = idx_sample - (idx_batch * self._sum_tree.time_dim_size)
        output = {k:self.buffers[k].buffer[idx_batch,...,idx_time] for k in self.buffers}

        # compute beta
        self._counter_for_beta += 1
        if beta is not None:
            # beta has been passed directly to this sample function, so use that
            pass
        elif self._n_beta_annealing_steps is not None:
            beta = min(1.,self._counter_for_beta / float(self._n_beta_annealing_steps))
        else:
            beta = 1.

        # sampled inverse priorities
        ipr_sample = 1./self._sum_tree[idx_batch,idx_time]

        # importance weights
        N = self._sum_tree.size
        pr_sum = self._sum_tree.sum()
        # w = q/p
        # q = 1/N
        # p ~ sum tree sampling = priority / sum(priorities)
        w = ipr_sample * pr_sum / N # scaled so that E_p[w] = 1 

        # clip large importance weights
        if self._wclip is not None:
            w = np.minimum(w,self._wclip)

        # annealing
        w = w ** beta

        assert 'importance_weight' not in output
        output['importance_weight'] = w

        self._idx_batch = idx_batch
        self._idx_time = idx_time
        self._idx_sample = idx_sample

        return output

    def update_priorities(self,priority):
        p = self._smooth_and_warp_priority(priority)
        unique_indices = np.unique(self._idx_sample) # to update inv_sum, need unique values
        dropped = self._sum_tree._flat_base[unique_indices]
        self._sum_tree[self._idx_batch,self._idx_time] = p
        p = self._sum_tree._flat_base[unique_indices]
        self._inv_sum += (1./p).sum() - (1./dropped).sum()
        self._idx_batch = None
        self._idx_time = None

    def priorities(self):
        return self._sum_tree

