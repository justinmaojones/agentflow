import gym
import numpy as np
from .base_env import BaseEnv

class GymEnv(BaseEnv):

    def __init__(self,env_id,noops=30,skip=4):
        self.env_id = env_id
        self.env = gym.make(env_id)
        self.noops = noops
        self.fire_reset = 'FIRE' in self.env.unwrapped.get_action_meanings()
        if skip > 1:
            assert 'NoFrameskip' in self.env.spec.id
        self.skip = skip

    def reset(self):
        ob = self.env.reset()
        self._obs_buf = np.stack([np.zeros_like(ob)]*2)
        for i in range(self.noops):
            ob, _, done, _ = self.env.step(0)
            if done:
                ob = self.env.reset()
        if self.fire_reset:
            ob, _, done, _ = self.env.step(1)
            if done:
                ob = self.env.reset()
            ob, _, done, _ = self.env.step(2)
            if done:
                ob = self.env.reset()
        return ob

    def step(self, action):
        total_reward = 0
        for i in range(self.skip):
            ob, reward, done, info = self.env.step(action)
            total_reward += reward
            if i >= self.skip - 2:
                self._obs_buf[i - (self.skip - 2)] = ob
            if done:
                break
        if done:
            ob = self.reset()
        else:
            ob = self._obs_buf.max(axis=0)
        return ob, total_reward, done, info

    @property
    def action_space(self):
        return self.env.action_space

    def n_actions(self):
        return self.action_space.n

    def action_shape(self):
        return self.env.action_space.shape

class VecGymEnv(BaseEnv):

    def __init__(self,env_id,n_envs=4,noops=30,skip=4):
        self.env_id = env_id
        self.n_envs = n_envs
        self.envs = [GymEnv(env_id,noops,skip) for i in range(n_envs)]

    def reset(self):
        return {'state': np.stack([env.reset() for env in self.envs])}

    def step(self,action):
        assert len(action) == len(self.envs), '%d %d'%(len(action),len(self.envs))
        obs, rewards, dones, infos = zip(*[env.step(a) for a,env in zip(action,self.envs)])
        return {
            'state': np.stack(obs), 
            'reward': np.stack(rewards), 
            'done': np.stack(dones),
            'info': infos,
        }

    def action_space(self):
        return self.envs[0].action_space

    def n_actions(self):
        return self.action_space().n

    def action_shape(self):
        return tuple([self.n_envs]+list(self.envs[0].action_space.shape))
