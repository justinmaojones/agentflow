from dataclasses import dataclass
import numpy as np

from agentflow.env.flow import EnvFlow


@dataclass
class TanhActionEnv(EnvFlow):

    scale: float = 1.0

    def step(self, action):
        transformed_action = self.scale * np.tanh(action)
        return self.source.step(transformed_action)
