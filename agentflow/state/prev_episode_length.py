import numpy as np
from ..env.base_env import BaseEnv

class PrevEpisodeLengthsEnv(BaseEnv):

    def __init__(self, env):
        self.env = env

    def n_actions(self):
        return self.env.n_actions()

    def action_shape(self):
        return self.env.action_shape()

    def reset(self):
        self._curr_episode_lengths = None
        self._prev_episode_lengths = None
        return self.env.reset()

    def step(self, action):
        output = self.env.step(action)
        if self._curr_episode_lengths is None:
            self._curr_episode_lengths = np.ones_like(output['reward']) 
            self._prev_episode_lengths = np.zeros_like(output['reward'])
        else:
            self._curr_episode_lengths += 1 
        self._prev_episode_lengths = output['done']*self._curr_episode_lengths + (1-output['done'])*self._prev_episode_lengths
        self._curr_episode_lengths = output['done']*np.zeros_like(output['reward']) + (1-output['done'])*self._curr_episode_lengths
        output['prev_episode_length'] = self._prev_episode_lengths
        return output



