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

    def __init__(self,
            max_length = 2**20,
            alpha = 0.6,
            eps = 1e-4,
            wclip = 32.,
            default_priority = 1.0,
            default_non_zero_reward_priority = None,
            default_done_priority = None,
            priority_key = None
            ):

        super(PrioritizedBufferMap,self).__init__(max_length)

        assert alpha >= 0
        self._alpha = alpha
        self._eps = eps
        self._wclip = wclip
        self._default_priority = default_priority
        self._default_non_zero_reward_priority = default_non_zero_reward_priority
        self._default_done_priority = default_done_priority

        self._idx_sample = None 
        self._sumtree = None 
        self._priority_key = priority_key

    def _compute_default_priority(self, data):
        priority = self._default_priority*np.ones_like(data['reward'], dtype=float)

        if self._default_non_zero_reward_priority is not None:
            priority[data['reward']!=0] = self._default_non_zero_reward_priority 

        if self._default_done_priority is not None:
            priority[data['done']==1] = self._default_done_priority

        return priority

    def _smooth_and_warp_priority(self,priority):
        return (np.abs(priority)+self._eps)**self._alpha

    def append(self, data, priority=None):
        if priority is None:
            if self._priority_key is not None:
                priority = data.pop(self._priority_key)
        else:
            assert self._priority_key is None, "cannot supply priority when priority key already set, instead provide priority in data"

        super(PrioritizedBufferMap,self).append(data)

        if priority is None:
            priority = self._compute_default_priority(data)

        if self._sumtree is None:
            self._sumtree = PrioritizedSamplingBuffer(self._max_length)

        p = self._smooth_and_warp_priority(priority)
        self._sumtree.append(p)

    def append_sequence(self,data,priority=None):
        super(PrioritizedBufferMap,self).append_sequence(data)

        if priority is None:
            t = list(data.values())[0].shape[-1]
            priority = np.ones((t,self.first_dim_size),dtype=float)

        if self._sumtree is None:
            self._sumtree = PrioritizedSamplingBuffer(self._max_length)

        p = self._smooth_and_warp_priority(priority)
        self._sumtree.append_sequence(p)

        assert self._index == self._sumtree._index, "index mismatch (%d, %d)" % (self._index, self._sumtree._index)

    def extend(self,X,priorities):
        for x,p in zip(X,priorities):
            self.append(x,p)

    def sample(self,nsamples,beta=None,normalized=True,with_indices=False):

        assert self._sumtree is not None, "cannot sample without first appending"

        if beta is None:
            beta = 1.

        idx_sample = self._sumtree.sample(nsamples)
        output = {k:self._buffers[k].get(*idx_sample) for k in self._buffers}

        # sampled inverse priorities
        ipr_sample = 1./self._sumtree.get(*idx_sample)

        # importance weights
        N = self._sumtree.size
        pr_sum = self._sumtree.sum()
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
        if with_indices:
            assert 'indices' not in output, "output cannot already contain key 'indices'"
            output['indices'] = idx_sample

        return output

    def update_priorities(self,priorities,indices=None):
        idx = indices if indices is not None else self._idx_sample
        if idx is None:
            raise ValueError("update_priorities must be called after sample or indices provided")
        p = self._smooth_and_warp_priority(priorities)
        self._sumtree[idx] = p
        self._idx_sample = None 

    def priorities(self):
        return self._sumtree
