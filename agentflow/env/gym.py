import gym
import numpy as np
from .base_env import BaseEnv

class VecGymEnv(BaseEnv):

    def __init__(self,env_id,n_envs=4):
        self.env_id = env_id
        self.n_envs = n_envs
        self.envs = [gym.make(env_id) for i in range(n_envs)]

    def reset(self):
        return np.stack([env.reset() for env in self.envs])

    def step(self,action):
        assert len(action) == len(self.envs), '%d %d'%(len(action),len(self.envs))
        obs, rewards, dones, infos = zip(*[env.step(a) for a,env in zip(action,self.envs)])
        obs = list(obs)
        for i,(done,env) in enumerate(zip(dones,self.envs)):
            if done:
                obs[i] = env.reset()
        return np.stack(obs), np.stack(rewards), np.stack(dones), infos

    def action_space(self):
        return self.envs[0].action_space

    def n_actions(self):
        return self.action_space().n

    def action_shape(self):
        return tuple([self.n_envs]+list(self.envs[0].action_space.shape))
