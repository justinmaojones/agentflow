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

    def update(self, frame, reset_mask=None):
        if frame.ndim != 4:
            raise ValueError(
                f"input to CropImageState must be 4d, received {frame.ndim} dims"
            )
        self._state = frame[:, self.top: -self.bottom, self.left: -self.right]
        return self.state()


class CropImageStateEnv(StatefulEnvFlow):
    def __init__(self, source, **kwargs):
        state = CropImageState(**kwargs)
        super(CropImageStateEnv, self).__init__(source, state)
