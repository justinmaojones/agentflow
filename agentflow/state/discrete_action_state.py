import numpy as np
from .state_env import StateEnv

class DiscreteActionEnv(StateEnv):

    def __init__(self,env,axis=-1,**kwargs):
        self.env = env
        self.axis = axis

    def reset(self):
        return self.env.reset()

    def step(self,action,*args,**kwargs):
        discrete_action = action.argmax(axis=self.axis) 
        return self.env.step(discrete_action,*args,**kwargs)

    def get_state(self):
        return self.env.get_state()
