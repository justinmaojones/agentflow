from dataclasses import dataclass
import numpy as np

from agentflow.env.flow import EnvFlow

@dataclass
class ClippedRewardEnv(EnvFlow):

    lower_bound: float
    upper_bound: float

    def step(self, action):
        output = self.source.step(action)
        output['reward'] = np.maximum(self.lower_bound, np.minimum(self.upper_bound, output['reward']))
        return output

