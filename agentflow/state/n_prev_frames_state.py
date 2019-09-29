import numpy as np

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
    return np.zeros(shape)

class NPrevFramesState(object):

    def __init__(self,n_prev_frames=4,to_dense=False):
        self.n_prev_frames = n_prev_frames
        self.to_dense = to_dense
        self.reset()

    def reset(self):
        self._state = None

    def update(self,frame,reset_mask=None,reset_val=0):

        # construct state ndarray using frame as a template
        if self._state is None:
            self._state = create_empty_state(frame,self.n_prev_frames)

        # reset state when reset_mask = 1
        if reset_mask is not None:
            self._state[reset_mask==1] = reset_val

        # update state
        shift_and_update_state(self._state,frame)

        return self.state()

    def state(self):
        if self.to_dense:
            first_dim_size = self._state.shape[0]
            return self._state.reshape(first_dim_size,-1)
        else:
            return self._state
