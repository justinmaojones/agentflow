from dataclasses import dataclass
import numpy as np

from agentflow.state.flow import State
from agentflow.state.flow import StatefulEnvFlow

@dataclass
class RandomOneHotMask(State):

    depth: int

    def _update_mask(self, indices):
        # indices along first dimension to update
        indices = np.array(indices)
        assert self._state is not None, "have you called env.reset() yet?"
        assert indices.ndim == 1, "indices must be a 1d array"
        self._state[indices] = False 
        rnd_idx = np.random.choice(self.depth, size=len(indices))
        self._state[indices, rnd_idx] = True

    def update(self, frame, reset_mask=None):
        if self._state is None:
            shape = (len(frame), self.depth)
            self._state = np.zeros(shape, dtype=bool)
            self._update_mask(np.arange(len(self._state)))
        else:
            if reset_mask is not None and sum(reset_mask) > 0:
                assert len(reset_mask) == len(self._state)
                # indices along first dim that need to be reset
                idx = np.arange(len(reset_mask))[reset_mask==1]
                self._update_mask(idx)
        return self._state.copy()

    def state(self):
        if self._state is not None:
            return self._state.copy()
        return None

class RandomOneHotMaskEnv(StatefulEnvFlow):

    def __init__(self, source, depth: int):
        self.source = source
        self.state = RandomOneHotMask(depth)

    def reset(self):
        prior_output = self.source.reset()
        self.state.reset()
        output = {k: prior_output[k] for k in prior_output}
        output['mask'] = self.state.update(prior_output['state'])
        return output

    def step(self, action):
        output = self.source.step(action)
        output['mask'] = self.state.update(output['state'], output['done'])
        return output
