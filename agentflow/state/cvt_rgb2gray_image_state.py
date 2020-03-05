import numpy as np
import cv2

class CvtRGB2GrayImageState(object):

    def __init__(self,flatten=False):
        self.flatten = flatten
        self.reset()

    def reset(self,frame=None,**kwargs):
        self._state = None

    def update(self,frame):
        n = len(frame)
        self._state = np.concatenate(
            [cv2.cvtColor(frame[i],cv2.COLOR_RGB2GRAY)[None,:,:,None] for i in range(n)],
            axis=0
        )
        return self.state()

    def state(self):
        if self.flatten:
            first_dim_size = self._state.shape[0]
            output = self._state.reshape(first_dim_size,-1)
        else:
            output = self._state
        return output.copy()

class CvtRGB2GrayImageStateEnv(object):

    def __init__(self,env,**kwargs):
        self.state = CvtRGB2GrayImageState(**kwargs)
        self.env = env

    def reset(self):
        frame = self.env.reset()
        self.state.reset()
        return self.state.update(frame)

    def step(self,*args,**kwargs):
        frame, reward, done, info = self.env.step(*args,**kwargs)
        return self.state.update(frame), reward, done, info

    def get_state(self):
        return self.state.state()

    def action_shape(self):
        return self.env.action_shape()
