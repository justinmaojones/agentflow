import numpy as np

from .base_env import BaseEnv
from ..numpy.ops import onehot

class ChainEnv(BaseEnv):
    """Implements the deterministic chain environment described in [1]

    References:
    [1] Osband, Ian, John Aslanides, and Albin Cassirer. "Randomized prior functions for 
        deep reinforcement learning." Advances in Neural Information Processing Systems. 2018.
    """

    def __init__(self, n_envs=4, length=8, action_mask=None):
        self.length = length
        self.n_envs = n_envs
        self.max_steps = length
        self.left_reward = 0
        self.right_reward = -0.01/self.max_steps
        self.right_most_reward = 1.

        if action_mask is None:
            self.action_mask = np.ones(self.length, dtype=int)
            self.action_mask = np.random.choice(2, size=(self.length, self.length))
        else:
            self.action_mask = action_mask
        self.reset()

    def copy(self, n_envs=None):
        n_envs = self.n_envs if n_envs is None else n_envs
        return ChainEnv(n_envs, self.length, self.action_mask)

    def _update_state(self):
        self._state = np.concatenate([
            #onehot(self._time*np.ones(self.n_envs, dtype=int), self.length),
            onehot(self._position, self.length)
        ],axis=-1)
        return self._state

    def _compute_reward(self, step):
        reward = (1-step)*self.left_reward + step*self.right_reward
        reward[self._position == self.length] = self.right_most_reward
        return reward

    def _compute_step(self, action):
        direction = self.action_mask[self._time, self._position]
        #direction = self.action_mask[self._position]
        # when direction is 1, don't change action
        # when direction is 0, flip action
        return action*direction + (1-action)*(1-direction)

    def _validate_action(self, action):
        assert action.shape == (self.n_envs,)
        assert len(set(action).difference(set([0,1]))) == 0

    def reset(self):
        self._time = 0 
        self._position = np.zeros(self.n_envs, dtype=int) 
        return {'state': self._update_state()}

    def step(self, action):
        self._validate_action(action)
        step = self._compute_step(action)
        self._position = np.maximum(0, self._position + 2*step - 1)
        self._time += 1
        rewards = self._compute_reward(step)
        if self._time == self.length:
            done = np.ones(self.n_envs)
            self.reset()
        else:
            done = np.zeros(self.n_envs)
        return {
            'state': self._update_state(), 
            'reward': rewards, 
            'done': done, 
            'info': {},
            'position': self._position,
            'time': self._time
        }

    def action_shape(self):
        return (self.n_envs, )

    def get_state(self):
        return self._state

    def n_actions(self):
        return 2
