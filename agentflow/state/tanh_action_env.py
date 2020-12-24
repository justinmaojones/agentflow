import numpy as np

from agentflow.env.base_env import BaseEnv

class TanhActionEnv(BaseEnv):

    def __init__(self, env, scale=1):
        self.env = env
        self.scale = scale

    def n_actions(self):
        return self.env.n_actions()

    def action_shape(self):
        return self.env.action_shape()

    def reset(self):
        return self.env.reset()

    def step(self, action):
        transformed_action = self.scale * np.tanh(action)
        return self.env.step(transformed_action)

