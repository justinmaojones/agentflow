import numpy as np
from starr import SumTreeArray

from .nd_array_buffer import NDArrayBuffer
from .buffer_map import BufferMap

class PrioritizedSamplingBuffer(NDArrayBuffer):

    def _build_buffer(self, shape, dtype):
        self._buffer = SumTreeArray(shape, dtype=dtype)

    def sample(self, n=1):
        if not (n > 0):
            raise ValueError("sample size must be greater than 0")

        if not (len(self) > 0):
            raise ValueError("cannot sample from buffer if it has no data appended")

        return self._buffer.sample(n)

    def sum(self):
        return self._buffer.sum()

class PrioritizedBufferMap(BufferMap):

    def __init__(self,max_length=2**20,alpha=0.6,eps=1e-4,wclip=32.,n_beta_annealing_steps=None):
        super(PrioritizedBufferMap,self).__init__(max_length)

        assert alpha >= 0
        self._alpha = alpha
        self._eps = eps
        self._wclip = wclip
        self._n_beta_annealing_steps = n_beta_annealing_steps
        self._counter_for_beta = 0

        self._idx_sample = None 
        self._sum_tree = None 

    def _smooth_and_warp_priority(self,priority):
        p = (np.abs(priority)+self._eps)**self._alpha
        return p

    def append(self,data,priority=None):
        super(PrioritizedBufferMap,self).append(data)

        if priority is None:
            priority = np.ones((self.first_dim_size,1),dtype=float)

        if self._sum_tree is None:
            self._sum_tree = PrioritizedSamplingBuffer(self._max_length)

        p = self._smooth_and_warp_priority(priority)
        self._sum_tree.append(p)

    def append_sequence(self,data,priority=None):
        super(PrioritizedBufferMap,self).append_sequence(data)

        if priority is None:
            t = list(data.values())[0].shape[-1]
            priority = np.ones((self.first_dim_size,t),dtype=float)

        if self._sum_tree is None:
            self._sum_tree = PrioritizedSamplingBuffer(self._max_length)

        p = self._smooth_and_warp_priority(priority)
        self._sum_tree.append_sequence(p)

        assert self._index == self._sum_tree._index, "index mismatch (%d, %d)" % (self._index, self._sum_tree._index)

    def extend(self,X,priorities):
        for x,p in zip(X,priorities):
            self.append(x,p)

    def sample(self,nsamples,beta=None,normalized=True):
        idx_sample = self._sum_tree.sample(nsamples)
        output = {k:self._buffers[k].get(*idx_sample) for k in self._buffers}

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
        ipr_sample = 1./self._sum_tree.get(*idx_sample)

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
        if normalized:
            output['importance_weight'] = w / w.mean()
        else:
            output['importance_weight'] = w

        self._idx_sample = idx_sample

        return output

    def update_priorities(self,priority):
        if self._idx_sample is None:
            raise ValueError("update_priorities must be called after sample")
        p = self._smooth_and_warp_priority(priority)
        self._sum_tree[self._idx_sample] = p
        self._idx_sample = None 

    def priorities(self):
        return self._sum_tree

