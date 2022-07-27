import numpy as np

from agentflow.env.flow import EnvFlow


class PrevEpisodeReturnsEnv(EnvFlow):
    def __post_init__(self):
        self._validate_source_flow()
        super().__post_init__()

    def _validate_source_flow(self):
        source = self.source
        while hasattr(source, "source"):
            source = source.source
            if isinstance(source, self.__class__):
                raise ValueError(
                    f"Cannot have more than one {self.__class__.__name__} in source flow"
                )

    def reset(self):
        self._curr_episode_returns = None
        self._prev_episode_returns = None
        return self.source.reset()

    def step(self, action):
        output = self.source.step(action)
        if self._curr_episode_returns is None:
            self._curr_episode_returns = output["reward"].copy()
            self._prev_episode_returns = np.zeros_like(output["reward"])
        else:
            self._curr_episode_returns += output["reward"]
        self._prev_episode_returns = (
            output["done"] * self._curr_episode_returns
            + (1 - output["done"]) * self._prev_episode_returns
        )
        self._curr_episode_returns = (
            output["done"] * np.zeros_like(output["reward"])
            + (1 - output["done"]) * self._curr_episode_returns
        )
        output["episode_return"] = self._prev_episode_returns

        if self.log is not None:
            self.log.append(
                f"{self.__class__.__name__}/episode_return", self._prev_episode_returns
            )

        return output
