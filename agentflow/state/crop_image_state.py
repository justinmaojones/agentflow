import numpy as np

from .base_state import BaseState
from .state_env import StateEnv

class CropImageState(BaseState):

    def __init__(self,top=0,bottom=0,left=0,right=0,flatten=False):
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right
        self.flatten = flatten
        self.reset()

    def update(self, frame, reset_mask=None):
        _, h, w, _ = frame.shape
        top = self.top
        left = self.left
        bottom = h-self.bottom
        right = w-self.right
        self._state = frame[:,top:bottom,left:right]
        return self.state()

class CropImageStateEnv(StateEnv):

    def __init__(self,env,**kwargs):
        state = CropImageState(**kwargs)
        super(CropImageStateEnv,self).__init__(env, state)
