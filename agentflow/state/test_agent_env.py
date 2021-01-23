import numpy as np
from ..env.base_env import BaseEnv
from ..numpy.ops import eps_greedy_noise
from . import PrevEpisodeReturnsEnv
from . import PrevEpisodeLengthsEnv

class TestAgentEnv(BaseEnv):

    def __init__(self, env):
        env = PrevEpisodeReturnsEnv(env)
        env = PrevEpisodeLengthsEnv(env)
        self.env = env

    def n_actions(self):
        return self.env.n_actions()

    def action_shape(self):
        return self.env.action_shape()

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def test(self, agent, session=None, noise_scale=0.0):
        state = self.reset()['state']
        all_done = None
        while all_done is None or np.mean(all_done) < 1:
            if noise_scale > 0:
                action_probs = agent.act_probs(
                    state=state, 
                    session=session, 
                )
                action = eps_greedy_noise(action_probs, eps=noise_scale)
            else:
                agent_output = agent.act(
                    state=state, 
                    session=session, 
                )
                action = agent_output['action']
            step_output = self.step(action)
            state = step_output['state']
            done = step_output['done']
            if all_done is None:
                all_done = done.copy()
            else:
                all_done = np.maximum(done,all_done)
        output = {
            'return': step_output['prev_episode_return'],
            'length': step_output['prev_episode_length'],
        }
        return output





