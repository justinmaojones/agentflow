import numpy as np

class RewardForwardFilterEnv(object):

    def __init__(self,env,gamma=0.99,**kwargs):
        self.env = env
        self.gamma = gamma

    def reset(self):
        self.state = self.env.reset()
        self.reward = None
        return self.state

    def step(self,*args,**kwargs):
        self.state, reward, done, info = self.env.step(*args,**kwargs)

        if self.reward is None:
            self.reward = np.zeros_like(reward)

        self.reward = self.reward*self.gamma + reward

        return self.state, self.reward, done, info

    def get_state(self):
        return self.state

    def action_shape(self):
        return self.env.action_shape()

