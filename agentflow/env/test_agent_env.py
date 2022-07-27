import numpy as np

from agentflow.env import PrevEpisodeReturnsEnv
from agentflow.env import PrevEpisodeLengthsEnv
from agentflow.env.flow import EnvFlow


class TestAgentEnv(EnvFlow):
    def __post_init__(self):
        source = self.source
        source = PrevEpisodeReturnsEnv(source, self.log)
        source = PrevEpisodeLengthsEnv(source, self.log)
        self.source = source

    def test(self, agent):
        state = self.reset()["state"]
        all_done = None
        while all_done is None or np.mean(all_done) < 1:
            action = agent.act(state)
            step_output = self.step(action)
            state = step_output["state"]
            done = step_output["done"]
            if all_done is None:
                all_done = done.copy()
            else:
                all_done = np.maximum(done, all_done)
        output = {
            "return": step_output["episode_return"],
            "length": step_output["episode_length"],
        }
        return output
