import numpy as np
import agentflow.numpy.ops as np_ops
from .base_state import BaseState
from .state_env import StateEnv

class AddEpisodeTimeState(BaseState):

    def __init__(self, flatten=False, binarized=False):
        self.flatten = flatten
        self.binarized = binarized
        super(AddEpisodeTimeState, self).__init__()

    def reset(self, frame=None, **kwargs):
        self._time = None
        super(AddEpisodeTimeState, self).reset(frame)

    def update(self,frame,reset_mask=None):

        # construct state ndarray using frame as a template
        if self._state is None:
            shape = list(frame.shape)
            shape[-1] = 1
            self._time = np.zeros(shape,dtype='int32')
        else:
            self._time += 1

        # reset state when reset_mask = 1
        if reset_mask is not None:
            self._time[reset_mask==1,...,-1] = 0 

        if self.binarized:
            time_state = np_ops.binarize(self._time,32)
            self._state = np.concatenate([frame,time_state],axis=-1)
        else:
            time_state = np.log(1.+self._time,dtype='float32')
            self._state = np.concatenate([frame,time_state],axis=-1)

        return self.state()

    def state(self):
        if self.flatten:
            first_dim_size = self._state.shape[0]
            output = self._state.reshape(first_dim_size,-1)
        else:
            output = self._state
        return output.copy()

class AddEpisodeTimeStateEnv(StateEnv):

    def __init__(self, env, **kwargs):
        state = AddEpisodeTimeState(**kwargs)
        super(AddEpisodeTimeStateEnv,self).__init__(env, state)
