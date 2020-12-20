import numpy as np
from .gym import VecGymEnv 

class VecBinaryPendulumEnv(VecGymEnv):

    def __init__(self,reward_threshold=-0.5,n_envs=4):
        self.reward_threshold = reward_threshold
        super(VecBinaryPendulumEnv,self).__init__('Pendulum-v0',n_envs)

    def step(self,action):
        obs, rewards, dones, infos = super(VecBinaryPendulumEnv,self).step(action)
        rewards = 2*((rewards) >= self.reward_threshold).astype(float) - 1
        return {
            'state': obs, 
            'reward': rewards, 
            'done': dones, 
            'info': infos,
        }
