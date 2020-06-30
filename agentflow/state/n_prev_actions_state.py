import numpy as np
from .state_env import StateEnv
from .n_prev_frames_state import NPrevFramesState
from agentflow.numpy.ops import onehot

class NPrevActionsStateEnv(StateEnv):

    def __init__(self,env,to_one_hot=True,dtype=np.float32,**kwargs):
        self.state = NPrevFramesState(**kwargs)
        self.to_one_hot = to_one_hot
        self.dtype = dtype
        super(NPrevActionsStateEnv,self).__init__(env)

    def reset(self):
        frame = self.env.reset()
        action_shape = self.env.action_shape() 
        if self.to_one_hot:
            n_actions = self.env.n_actions()
            action_shape = action_shape + (n_actions,)
        self.state.reset()
        action_reset = np.zeros(action_shape,dtype=self.dtype)
        action_state = self.state.update(action_reset)
        self._state = {
            'frame': frame,
            'action': action_state,
        }
        return self._state

    def step(self,action,*args,**kwargs):
        frame, reward, done, info = self.env.step(action,*args,**kwargs)
        if self.to_one_hot:
            action = onehot(action,self.n_actions())
        action_state = self.state.update(action,done)
        self._state = {
            'frame': frame,
            'action': action_state,
        }
        return self._state, reward, done, info

    def get_state(self):
        return self.state.state()
