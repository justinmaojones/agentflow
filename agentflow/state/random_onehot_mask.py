import numpy as np

from agentflow.state.base_state import BaseState
from agentflow.env.base_env import BaseEnv

class RandomOneHotMask(BaseState):

    def __init__(self, dim):
        self.dim = dim
        self._state = None

    def _update_mask(self, indices):
        # indices along first dimension to update
        indices = np.array(indices)
        assert self._state is not None, "have you called env.reset() yet?"
        assert indices.ndim == 1, "indices must be a 1d array"
        self._state[indices] = False 
        rnd_idx = np.random.choice(self.dim, size=len(indices))
        self._state[indices, rnd_idx] = True

    def reset(self, frame):
        shape = (len(frame), self.dim)
        self._state = np.zeros(shape, dtype=np.bool)
        self._update_mask(np.arange(len(self._state)))

    def update(self, frame, reset_mask=None):
        if self._state is None:
            self.reset(frame)
        else:
            if reset_mask is not None and sum(reset_mask) > 0:
                assert len(reset_mask) == len(self._state)
                # indices along first dim that need to be reset
                idx = np.arange(len(reset_mask))[reset_mask==1]
                self._update_mask(idx)
        return self._state.copy()

    def state(self):
        return self._state.copy()

class RandomOneHotMaskEnv(BaseEnv):

    def __init__(self, env, dim):
        self.env = env
        self.state = RandomOneHotMask(dim)

    def n_actions(self):
        return self.env.n_actions()

    def action_shape(self):
        return self.env.action_shape()

    def reset(self):
        prior_output = self.env.reset()
        self.state.reset(prior_output['state'])
        output = {k: prior_output[k] for k in prior_output}
        output['mask'] = self.state.state()
        return output

    def step(self, action):
        output = self.env.step(action)
        output['mask'] = self.state.update(output['state'], output['done'])
        return output
