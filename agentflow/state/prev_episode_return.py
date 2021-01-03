import numpy as np
from ..env.base_env import BaseEnv

class PrevEpisodeReturnsEnv(BaseEnv):

    def __init__(self, env):
        self.env = env

    def n_actions(self):
        return self.env.n_actions()

    def action_shape(self):
        return self.env.action_shape()

    def reset(self):
        self._curr_episode_returns = None
        self._prev_episode_returns = None
        return self.env.reset()

    def step(self, action):
        output = self.env.step(action)
        if self._curr_episode_returns is None:
            self._curr_episode_returns = output['reward'].copy()
            self._prev_episode_returns = np.zeros_like(output['reward'])
        else:
            self._curr_episode_returns += output['reward']
        self._prev_episode_returns = output['done']*self._curr_episode_returns + (1-output['done'])*self._prev_episode_returns
        self._curr_episode_returns = output['done']*np.zeros_like(output['reward']) + (1-output['done'])*self._curr_episode_returns
        output['prev_episode_return'] = self._prev_episode_returns
        return output



