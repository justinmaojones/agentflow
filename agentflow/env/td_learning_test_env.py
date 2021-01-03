import numpy as np
import agentflow.numpy.ops as np_ops
from .base_env import BaseEnv

class TDLearningTestEnv(BaseEnv):

    def __init__(self,
            n_envs=1,
            episode_length=100,
            reward_freq=10,
            reward_multiplier=1,
            time_state=True,
            binarize_time_state=True,
            random_dims=10,
            static_dims=10,
        ):

        self.n_envs = n_envs
        self.episode_length = episode_length
        self.reward_freq = reward_freq
        self.reward_multiplier = reward_multiplier
        self.time_state = time_state
        self.binarize_time_state = binarize_time_state
        self.random_dims = random_dims
        self.static_dims = static_dims

        self._t = None
        self._static_state = None

    def _build_state(self):
        state = []

        if self.time_state:
            time_state = self._t*np.ones((self.n_envs,1))
            if self.binarize_time_state:
                base = int(np.ceil(np.log2(self.episode_length)))
                time_state = np_ops.binarize(time_state,base)
            else:
                time_state = time_state/self.episode_length - 0.5
            state.append(time_state)

        if self.random_dims > 0:
            random_state = np.random.randn(self.n_envs,self.random_dims)
            state.append(random_state)

        if self.static_dims > 0:
            if self._static_state is None:
                self._static_state = np.stack([np.random.randn(self.static_dims)]*self.n_envs)
                self._static_state = self._static_state/self._static_state.max() - 0.5
            state.append(self._static_state)

        self._state = np.concatenate(state,axis=-1)
        return self._state

    def reset(self):
        self._t = 0
        return {'state': self._build_state()}

    def _build_reward(self):
        if self._t % self.reward_freq == 0:
            return self.reward_multiplier*np.ones(self.n_envs)
        else:
            return np.zeros(self.n_envs)

    def _build_done(self):
        if self._t == 0:
            return np.ones(self.n_envs)
        else:
            return np.zeros(self.n_envs)

    def step(self,action):
        assert self._t is not None, "did you forget to reset?"
        self._t = (self._t + 1) % self.episode_length
        state = self._build_state()
        reward = self._build_reward()
        done = self._build_done() 
        return {
            'state': state, 
            'reward': reward, 
            'done': done, 
            'info': {},
        }

    def n_actions(self):
        return 1

    def action_shape(self):
        return (1,)
