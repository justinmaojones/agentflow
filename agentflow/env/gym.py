import gym
import numpy as np

from agentflow.env.source import EnvSource

class GymEnv(EnvSource):

    def __init__(self, env_id, noops=30, frames_per_action=1, fire_reset=False):
        self.env_id = env_id
        self.env = gym.make(env_id)
        self.noops = noops
        self.frames_per_action = frames_per_action
        self.fire_reset = fire_reset

    def _reshape_action(self, action):
        return np.reshape(action, self.action_space.shape)

    def reset(self):
        ob = self.env.reset()
        if self.fire_reset:
            ob, _, done, _ = self.env.step(self._reshape_action(1))
            if done:
                ob = self.env.reset()
            ob, _, done, _ = self.env.step(self._reshape_action(2))
            if done:
                ob = self.env.reset()
        for i in range(self.noops):
            ob, _, done, _ = self.env.step(self._reshape_action(0))
            if done:
                ob = self.env.reset()
        return ob

    def step(self, action):
        total_reward = 0
        for i in range(self.frames_per_action):
            ob, reward, done, info = self.env.step(self._reshape_action(action))
            total_reward += reward
            if done:
                break
        if done:
            ob = self.reset()
        return ob, total_reward, done, info

    @property
    def action_space(self):
        return self.env.action_space

    def n_actions(self):
        return self.action_space.n

    def action_shape(self):
        return self.env.action_space.shape

class VecGymEnv(EnvSource):

    def __init__(self, env_id, n_envs=4, noops=30, frames_per_action=4, fire_reset=False):
        self.env_id = env_id
        self.n_envs = n_envs
        self.noops = noops
        self.frames_per_action = frames_per_action
        self.envs = [GymEnv(env_id, noops, frames_per_action, fire_reset) for i in range(n_envs)]

    def reset(self):
        return {'state': np.stack([env.reset() for env in self.envs])}

    def step(self, action):
        assert len(action) == len(self.envs), '%d %d'%(len(action), len(self.envs))
        obs, rewards, dones, infos = zip(*[env.step(a) for a, env in zip(action, self.envs)])
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

class CartpoleGymEnv(VecGymEnv):
    def __init__(self, n_envs=1):
        super().__init__('CartPole-v1', n_envs, noops=0, frames_per_action=1)

class PendulumGymEnv(VecGymEnv):
    def __init__(self, n_envs=1):
        super().__init__('Pendulum-v1', n_envs, noops=0, frames_per_action=1)

class AtariGymEnv(VecGymEnv):
    def __init__(self, env_id, n_envs=1, frames_per_action=1, fire_reset=False):
        super().__init__(env_id, n_envs, noops=30, frames_per_action=frames_per_action, fire_reset=fire_reset)
