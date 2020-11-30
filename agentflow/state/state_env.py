from agentflow.env.base_env import BaseEnv

class StateEnv(BaseEnv):

    def __init__(self, env, state):
        self.env = env
        self.state = state

    def n_actions(self):
        return self.env.n_actions()

    def action_shape(self):
        return self.env.action_shape()

    def reset(self):
        frame = self.env.reset()
        self.state.reset()
        return self.state.update(frame)

    def step(self,*args,**kwargs):
        frame, reward, done, info = self.env.step(*args,**kwargs)
        return self.state.update(frame,done), reward, done, info

    def get_state(self):
        return self.state.state()

