from dataclasses import dataclass

import numpy as np
import cv2
from typing import Tuple

from agentflow.state.flow import State
from agentflow.state.flow import StatefulEnvFlow


@dataclass
class ResizeImageState(State):

    resized_shape: Tuple[int]
    flatten: bool = False

    def update(self, frame, reset_mask=None):
        n = len(frame)
        self._state = np.concatenate(
            [
                cv2.resize(frame[i], self.resized_shape, interpolation=cv2.INTER_AREA)[
                    None
                ]
                for i in range(n)
            ],
            axis=0,
        )
        while len(self._state.shape) < len(frame.shape):
            self._state = self._state[..., None]
        return self.state()

    def state(self):
        if self.flatten:
            first_dim_size = self._state.shape[0]
            output = self._state.reshape(first_dim_size, -1)
        else:
            output = self._state
        return output.copy()


class ResizeImageStateEnv(StatefulEnvFlow):
    def __init__(self, source, *args, **kwargs):
        state = ResizeImageState(*args, **kwargs)
        super(ResizeImageStateEnv, self).__init__(source, state)
