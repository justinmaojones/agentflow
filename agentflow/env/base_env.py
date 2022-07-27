from agentflow.logging import WithLogging


class BaseEnv(WithLogging):
    def reset(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def n_actions(self):
        raise NotImplementedError

    def action_shape(self):
        raise NotImplementedError
