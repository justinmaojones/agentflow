import numpy as np
from .base_state import BaseState
from .state_env import StateEnv

def shift_and_update_state(state,frame):
    ndim = state.ndim
    T = state.shape[-1]
    idx_prev = tuple([slice(None)]*(ndim-1) + [slice(0,T-1)])
    idx_next = tuple([slice(None)]*(ndim-1) + [slice(1,T)])
    idx_frame = tuple([slice(None)]*(ndim-1) + [0])
    state[idx_next] = state[idx_prev]
    state[idx_frame] = frame
    return state

def create_empty_state(frame,n_prev_frames):
    shape = list(frame.shape) + [n_prev_frames]
    return np.zeros(shape,dtype=frame.dtype)

class NPrevFramesState(BaseState):

    def __init__(self,n_prev_frames=4,flatten=False):
        self.n_prev_frames = n_prev_frames
        self.flatten = flatten
        super(AddEpisodeTimeState, self).__init__()

    def reset(self,frame=None,**kwargs):
        self._new_shape = None
        super(AddEpisodeTimeState, self).reset(frame)

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
        shift_and_update_state(self._state,frame)

        return self.state()

    def state(self):
        if self.flatten:
            output = self._state.reshape(*self._new_shape)
        else:
            output = self._state
        return output.copy()

class NPrevFramesStateEnv(StateEnv):

    def __init__(self,env,**kwargs):
        state = NPrevFramesState(**kwargs)
        super(NPrevFramesStateEnv,self).__init__(env state)
