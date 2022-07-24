from dataclasses import dataclass
import numpy as np

from agentflow.state.flow import State
from agentflow.state.flow import StatefulEnvFlow


@dataclass
class CropImageState(State):

    top: int = 0
    bottom: int = 0
    left: int = 0
    right: int = 0
    flatten: bool = False

    def update(self, frame, reset_mask=None):
        _, h, w, _ = frame.shape
        top = self.top
        left = self.left
        bottom = h-self.bottom
        right = w-self.right
        self._state = frame[:,top:bottom,left:right]
        return self.state()

class CropImageStateEnv(StatefulEnvFlow):

    def __init__(self, source, **kwargs):
        state = CropImageState(**kwargs)
        super(CropImageStateEnv,self).__init__(source, state)
