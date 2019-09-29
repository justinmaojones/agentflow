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

    def __init__(self,n_prev_frames=4,flatten=False):
        self.n_prev_frames = n_prev_frames
        self.flatten = flatten
        self.reset()

    def reset(self,frame=None,**kwargs):
        self._state = None

    def update(self,frame,reset_mask=None):

        # construct state ndarray using frame as a template
        if self._state is None:
            self._state = create_empty_state(frame,self.n_prev_frames)

        # reset state when reset_mask = 1
        if reset_mask is not None:
            self._state[reset_mask==1] = 0 

        # update state
        shift_and_update_state(self._state,frame)

        return self.state()

    def state(self):
        if self.flatten:
            first_dim_size = self._state.shape[0]
            return self._state.reshape(first_dim_size,-1)
        else:
            return self._state

class NPrevFramesStateEnv(object):

    def __init__(self,env,**kwargs):
        self.state = NPrevFramesState(**kwargs)
        self.env = env

    def reset(self):
        frame = self.env.reset()
        self.state.reset()
        return self.state.update(frame)

    def step(self,*args,**kwargs):
        frame, reward, done, info = self.env.step(*args,**kwargs)
        return self.state.update(frame), reward, done, info
