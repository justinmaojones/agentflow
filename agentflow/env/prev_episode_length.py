import numpy as np

from agentflow.env.flow import EnvFlow

class PrevEpisodeLengthsEnv(EnvFlow):

    def __post_init__(self):
        self._validate_source_flow()
        super().__post_init__()

    def _validate_source_flow(self):
        source = self.source
        while hasattr(source, 'source'):
            source = source.source
            if isinstance(source, self.__class__):
                raise ValueError(f"Cannot have more than one {self.__class__.__name__} in source flow")

    def reset(self):
        self._curr_episode_lengths = None
        self._prev_episode_lengths = None
        return self.source.reset()

    def step(self, action):
        output = self.source.step(action)
        if self._curr_episode_lengths is None:
            self._curr_episode_lengths = np.ones_like(output['reward']) 
            self._prev_episode_lengths = np.zeros_like(output['reward'])
        else:
            self._curr_episode_lengths += 1 
        self._prev_episode_lengths = output['done']*self._curr_episode_lengths + (1-output['done'])*self._prev_episode_lengths
        self._curr_episode_lengths = output['done']*np.zeros_like(output['reward']) + (1-output['done'])*self._curr_episode_lengths
        output['episode_length'] = self._prev_episode_lengths

        if self.log is not None:
            self.log.append(f"{self.__class__.__name__}/episode_length", self._prev_episode_lengths)

        return output



