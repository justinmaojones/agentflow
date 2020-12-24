import numpy as np

def create_empty_state(frame,n_prev_frames):
    shape = list(frame.shape) + [n_prev_frames]
    return np.zeros(shape,dtype=frame.dtype)

def shift_and_update_state(state,frame):
    ndim = state.ndim
    T = state.shape[-1]
    idx_prev = tuple([slice(None)]*(ndim-1) + [slice(0,T-1)])
    idx_next = tuple([slice(None)]*(ndim-1) + [slice(1,T)])
    idx_frame = tuple([slice(None)]*(ndim-1) + [0])
    state[idx_next] = state[idx_prev]
    state[idx_frame] = frame
    return state
