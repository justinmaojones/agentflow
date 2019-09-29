

class LastNFrames(object):

    def __init__(self,n_prev=4):
        self.n_prev = n_prev
        self.state = None

    def preprocess(self,frame,done=None):
        if self.state is None:
            # frame.shape: n_envs x [frame_dims]
            # self.state.shape: n_envs x [frame_dims] x self.n_prev
            shape = list(frame.shape) + [self.n_prev]
            self.state = np.zeros(shape)

        if done is not None:
            # done.shape: n_envs
            self.state[done==1] = 0

        ndim = self.state.ndim
        idx_prev = [slice(None)]*(ndim-1) + [slice(0,self.n_prev-1)]
        idx_next = [slice(None)]*(ndim-1) + [slice(1,self.n_prev)]
        self.state[idx_next] = self.state[idx_prev]

        idx_new = [slice(None)]*(ndim-1) + [slice(0)]
        self.state[idx_new] = frame

        return self.state
