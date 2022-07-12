import numpy as np
from ..env.base_env import BaseEnv
from . import PrevEpisodeReturnsEnv
from . import PrevEpisodeLengthsEnv

from agentflow.logging import LogsTFSummary

class TestAgentEnv(BaseEnv):

    def __init__(self, env: BaseEnv, log: LogsTFSummary = None):
        env = PrevEpisodeReturnsEnv(env, log)
        env = PrevEpisodeLengthsEnv(env, log)
        self.env = env
        self.log = log

    def n_actions(self):
        return self.env.n_actions()

    def action_shape(self):
        return self.env.action_shape()

    def reset(self):
        return self.env.reset()

    # TODO: added because env has not been migrated to source/flow
    def set_log(self, log: LogsTFSummary):
        super().set_log(log)
        self.env.set_log(log)

    def step(self, action):
        return self.env.step(action)

    def test(self, agent):
        state = self.reset()['state']
        all_done = None
        while all_done is None or np.mean(all_done) < 1:
            action = agent.act(state)
            step_output = self.step(action)
            state = step_output['state']
            done = step_output['done']
            if all_done is None:
                all_done = done.copy()
            else:
                all_done = np.maximum(done,all_done)
        output = {
            'return': step_output['episode_return'],
            'length': step_output['episode_length'],
        }
        return output





