class BaseState(object):

    def __init__(self):
        self.reset()

    def reset(self, frame=None, **kwargs):
        self._state = frame

    def update(self, frame, reset_mask=None):
        raise NotImplementedError

    def state(self):
        return self._state


