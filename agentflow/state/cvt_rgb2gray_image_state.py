from dataclasses import dataclass
import numpy as np
import cv2

from agentflow.state.flow import State
from agentflow.state.flow import StatefulEnvFlow

@dataclass
class CvtRGB2GrayImageState(State):

    flatten: bool = False

    def update(self, frame, reset_mask=None):
        n = len(frame)
        self._state = np.concatenate(
            [cv2.cvtColor(frame[i],cv2.COLOR_RGB2GRAY)[None,:,:,None] for i in range(n)],
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

class CvtRGB2GrayImageStateEnv(StatefulEnvFlow):

    def __init__(self, source, **kwargs):
        state = CvtRGB2GrayImageState(**kwargs)
        super(CvtRGB2GrayImageStateEnv, self).__init__(source, state)

