import numpy as np
from ..env.base_env import BaseEnv

from agentflow.logging import LogsTFSummary


class PrevEpisodeLengthsEnv(BaseEnv):

    def __init__(self, env: BaseEnv, log: LogsTFSummary = None):
        self.env = env
        self.log = log

        self._validate_env_flow()

    def _validate_env_flow(self):
        env = self.env
        while hasattr(env, 'env'):
            env = env.env
            if isinstance(env, self.__class__):
                raise ValueError(f"Cannot have more than one {self.__class__.__name__} in env flow")

    def n_actions(self):
        return self.env.n_actions()

    def action_shape(self):
        return self.env.action_shape()

    def reset(self):
        self._curr_episode_lengths = None
        self._prev_episode_lengths = None
        return self.env.reset()

    # TODO: added because env has not been migrated to source/flow
    def set_log(self, log: LogsTFSummary):
        super().set_log(log)
        self.env.set_log(log)

    def step(self, action):
        output = self.env.step(action)
        if self._curr_episode_lengths is None:
            self._curr_episode_lengths = np.ones_like(output['reward']) 
            self._prev_episode_lengths = np.zeros_like(output['reward'])
        else:
            self._curr_episode_lengths += 1 
        self._prev_episode_lengths = output['done']*self._curr_episode_lengths + (1-output['done'])*self._prev_episode_lengths
        self._curr_episode_lengths = output['done']*np.zeros_like(output['reward']) + (1-output['done'])*self._curr_episode_lengths
        output['episode_length'] = self._prev_episode_lengths

        if self.log is not None:
            self.log.append(f"{self.__class__.__name__}/episode_length", self._prev_episode_lengths)

        return output



