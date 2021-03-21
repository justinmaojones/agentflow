import numpy as np

from agentflow.env.base_env import BaseEnv

class ClippedRewardEnv(BaseEnv):

    def __init__(self, env, lower_bound=-1, upper_bound=1):
        self.env = env
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def n_actions(self):
        return self.env.n_actions()

    def action_shape(self):
        return self.env.action_shape()

    def reset(self):
        return self.env.reset()

    def step(self, action):
        output = self.env.step(action)
        output['reward'] = np.maximum(self.lower_bound, np.minimum(self.upper_bound, output['reward']))
        return output

