from .base_state import BaseState
from .state_env import StateEnv

class IdentityState(BaseState):

    def update(self, frame, reset_mask=None):
        self._state = frame
        return self.state()

class IdentityStateEnv(StateEnv):

    def __init__(self,env,**kwargs):
        state = IdentityState(**kwargs)
        super(IdentityStateEnv,self).__init__(env, state)

