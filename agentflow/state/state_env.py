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
        prior_output = self.env.reset()
        self.state.reset()
        output = {k: prior_output[k] for k in prior_output if k != 'state'}
        output['state'] = self.state.update(prior_output['state'])
        return output

    def step(self,*args,**kwargs):
        prior_step_output = self.env.step(*args,**kwargs)
        # copy non-state data from previous step output
        output = {k: prior_step_output[k] for k in prior_step_output if k != 'state'}
        # update state
        output['state'] = self.state.update(prior_step_output['state'], prior_step_output['done'])
        return output

    def get_state(self):
        return self.state.state()

