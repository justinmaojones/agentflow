import numpy as np
import cv2

from .base_state import BaseState
from .state_env import StateEnv

class ResizeImageState(BaseState):

    def __init__(self,resized_shape,flatten=False):
        self.resized_shape = resized_shape
        self.flatten = flatten
        self.reset()

    def update(self,frame):
        n = len(frame)
        self._state = np.concatenate(
            [cv2.resize(frame[i],self.resized_shape)[None] for i in range(n)],
            axis=0
        )
        return self.state()

    def state(self):
        if self.flatten:
            first_dim_size = self._state.shape[0]
            output = self._state.reshape(first_dim_size,-1)
        else:
            output = self._state
        return output.copy()

class ResizeImageStateEnv(StateEnv):

    def __init__(self,env,**kwargs):
        state = ResizeImageState(**kwargs)
        super(ResizeImageStateEnv,self).__init__(env, state)
