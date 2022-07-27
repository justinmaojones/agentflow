from dataclasses import dataclass
import numpy as np

from agentflow.state.flow import State
from agentflow.state.flow import StatefulEnvFlow

def create_empty_state(frame, n_prev_frames):
    shape = list(frame.shape) + [n_prev_frames]
    return np.zeros(shape, dtype=frame.dtype)

def shift_and_update_state(state, frame):
    output = np.roll(state, 1, axis=-1)
    output[..., 0] = frame
    return output

@dataclass
class NPrevFramesState(State):

    n_prev_frames: int
    flatten: bool = False

    def reset(self):
        self._new_shape = None
        super().reset()

    def update(self,frame,reset_mask=None):

        # construct state ndarray using frame as a template
        if self._state is None:
            self._state = create_empty_state(frame,self.n_prev_frames)
            shape = self._state.shape
            self._new_shape = [s for s in shape[:-2]] + [shape[-2]*shape[-1]]

        # reset state when reset_mask = 1
        if reset_mask is not None:
            self._state[reset_mask==1] = 0 

        # update state
        self._state = shift_and_update_state(self._state, frame)

        return self.state()

    def state(self):
        if self.flatten:
            output = self._state.reshape(*self._new_shape)
        else:
            output = self._state
        return output.copy()

class NPrevFramesStateEnv(StatefulEnvFlow):

    def __init__(self, source, **kwargs):
        state = NPrevFramesState(**kwargs)
        super(NPrevFramesStateEnv,self).__init__(source, state)
