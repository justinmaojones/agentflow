import numpy as np

class AppendEpisodeTimeState(object):

    def __init__(self,log_transform=True):
        self.log_transform = log_transform
        self.reset()

    def reset(self,frame=None,**kwargs):
        self._state = None
        self._time = None

    def update(self,frame,reset_mask=None):

        # construct state ndarray using frame as a template
        if self._time is None:
            self._time = np.zeros(len(frame),dtype='int32')
        else:
            self._time += 1

        # reset state when reset_mask = 1
        if reset_mask is not None:
            self._time[reset_mask==1] = 0 

        self._frame = frame
        if self.log_transform:
            self._state = np.log(1.+self._time,dtype='float32')
        else:
            self._state = self._time

        return self.state()

    def state(self):
        return self._frame, self._state.copy()

class AppendEpisodeTimeStateEnv(object):

    def __init__(self,env,**kwargs):
        self.state = AppendEpisodeTimeState(**kwargs)
        self.env = env

    def reset(self):
        frame = self.env.reset()
        self.state.reset()
        frame, time_state = self.state.update(frame)
        return frame, time_state

    def step(self,*args,**kwargs):
        frame, reward, done, info = self.env.step(*args,**kwargs)
        _, time_state = self.state.update(frame,done)
        return frame, time_state, reward, done, info

    def get_state(self):
        return self.state.state()

    def action_shape(self):
        return self.env.action_shape()

