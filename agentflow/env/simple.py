import numpy as np

class VecConcaveFuncEnv(object):

    def __init__(self,n_dims=1,n_envs=4,max_steps=100,min_reward=-20):
        self.n_dims = n_dims
        self.n_envs = n_envs
        self._state = None
        self.max_steps = max_steps
        self.min_reward = min_reward
        self._steps = None

    def reset(self,done=None):
        reset_state = np.random.randn(self.n_envs,self.n_dims)
        reset_steps = np.zeros(self.n_envs)
        if done is None:
            self._state = reset_state
            self._steps = reset_steps
        else:
            done = done.reshape(-1,1)
            self._state = self._state*(1-done) + reset_state*done
            done = done.ravel()
            self._steps = self._steps*(1-done) + reset_steps*done
        return self._state

    def compute_reward(self,state):
        return -(state**2).sum(axis=1)**0.5

    def step(self,action):
        assert len(action) == self.n_envs
        action = action.reshape(self._state.shape)
        self._state = self._state + action
        self._steps += 1

        # negative root mean square
        rewards = self.compute_reward(self._state)

        dones = np.logical_or(
                self._steps >= self.max_steps,
                rewards <= self.min_reward).astype(float)

        self._state = self.reset(dones)

        return self._state, rewards, dones, {}

    def action_space(self):
        return None

    def action_shape(self):
        return (self.n_envs, self.n_dims)


