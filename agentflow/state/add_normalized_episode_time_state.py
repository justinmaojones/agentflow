import numpy as np

class AddNormalizedEpisodeTimeState(object):

    def __init__(self,flatten=False,ema_decay=0.99,scale=1,initial_estimate=1000):
        self.flatten = flatten
        self.ema_decay = ema_decay
        self.scale = scale
        self.initial_estimate = initial_estimate
        self._mean_time = self.initial_estimate
        self.reset()

    def reset(self,frame=None,**kwargs):
        self._state = None
        self._time = None

    def update(self,frame,reset_mask=None):

        # construct state ndarray using frame as a template
        if self._state is None:
            shape = list(frame.shape)
            shape[-1] = 1
            self._time = np.zeros(shape,dtype='int32')
        else:
            self._time += 1

        self._mean_time = self.ema_decay*self._mean_time + (1-self.ema_decay)*np.mean(self._time)

        # reset state when reset_mask = 1
        if reset_mask is not None:
            self._time[reset_mask==1,...,-1] = 0 

        time_state = np.log(1.+self._time)
        time_state = (self.scale*time_state/(1.+self._mean_time+time_state) ).astype('float32')
        self._state = np.concatenate([frame,time_state],axis=-1)

        return self.state()

    def state(self):
        if self.flatten:
            first_dim_size = self._state.shape[0]
            output = self._state.reshape(first_dim_size,-1)
        else:
            output = self._state
        return output.copy()

class AddNormalizedEpisodeTimeStateEnv(object):

    def __init__(self,env,**kwargs):
        self.state = AddNormalizedEpisodeTimeState(**kwargs)
        self.env = env

    def reset(self):
        frame = self.env.reset()
        self.state.reset()
        return self.state.update(frame)

    def step(self,*args,**kwargs):
        frame, reward, done, info = self.env.step(*args,**kwargs)
        return self.state.update(frame,done), reward, done, info

    def get_state(self):
        return self.state.state()

    def action_shape(self):
        return self.env.action_shape()

