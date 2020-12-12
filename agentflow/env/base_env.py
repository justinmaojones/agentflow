
class BaseEnv(object):

    def reset(self):
        raise NotImplementedError

    def step(self,action):
        raise NotImplementedError

    def step_dict(self,action):
        obs, rewards, dones, _ = self.step(action)
        return {
            'state': obs,
            'rewards': rewards,
            'dones': dones,
        }


    def n_actions(self):
        raise NotImplementedError

    def action_shape(self):
        raise NotImplementedError
