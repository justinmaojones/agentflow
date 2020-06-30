from agentflow.env.base_env import BaseEnv

class StateEnv(BaseEnv):

    def __init__(self,env):
        self.env = env

    def n_actions(self):
        return self.env.n_actions()

    def action_shape(self):
        return self.env.action_shape()
