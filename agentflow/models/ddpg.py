import tensorflow as tf
from trfl.dpg_ops import dpg
from .buffer import NDArrayBuffer 
from .ops import exponential_moving_average


class BufferMap(object):
    
    def __init__(self,max_length=1e6):
        self.max_length = max_length
        self._n = 0
        self.buffers = {
            'reward':NDArrayBuffer(self.max_length),
            'action':NDArrayBuffer(self.max_length),
            'state':NDArrayBuffer(self.max_length),
            'done':NDArrayBuffer(self.max_length),
        }

    def __len__(self):
        return self._n

    def append(self,data):
        for k in self.buffers:
            assert k in data
            self.buffers[k].append(data[k])
        self._n += 1

    def extend(self,X):
        for x in X:
            self.append(x)

    def sample(self,nsamples):
        assert nsamples <= self._n-1 #because there should always exist a state2
        idx = np.random.choice(self._n-1,size=nsamples,replace=False)
        output = {k:self.buffers[k].get(idx) for k in self.buffers}
        output['state2'] = self.buffers['state'].get(idx+1)
        return output

        
class DDPG(object):

    def __init__(self,state_shape,action_shape,policy_fn,q_fn,dqda_clipping=None,clip_norm=False):
        """Implements Deep Deterministic Policy Gradient with Tensorflow

        This class builds a DDPG model with optimization update and action prediction steps.

        Args:
          state_shape: a tuple or list of the state shape, excluding the batch dimension.
            For example, for images of size 28 x 28 x 3, state_shape=[28,28,3].
          action_shape: a tuple or list of the action shape, excluding the batch dimension.
            For example, for scalar actions, action_shape=[].  For a vector of actions
            with 3 elements, action_shape[3].
          policy_fn: a function that takes as input a tensor, the state, and
            outputs an action (with shape=action_shape, excluding batch dimension).
          q_fn: a function that takes as input two tensors: the state and action,
            and outputs an estimate Q(state,action)

        """
        self.state_shape = list(state_shape)
        self.action_shape = list(action_shape)
        self.policy_fn = policy_fn
        self.q_fn = q_fn
        self.dqda_clipping = dqda_clipping
        self.clip_norm = clip_norm
        self.build_model()

    def build_model(self):

        # inputs
        state = tf.placeholder(tf.float32,shape=tuple([None]+self.state_shape)
        action = tf.placeholder(tf.int32,shape=tuple([None]+self.action_shape)
        reward = tf.placeholder(tf.float32,shape=(None,))
        done = tf.placeholder(tf.float32,shape=(None,))
        state2 = tf.placeholder(tf.float32,shape=tuple([None]+self.state_shape))
        gamma = tf.placeholder(tf.float32)
        learning_rate = tf.placeholder(tf.float32)
        ema_decay = tf.placeholder(tf.float32)

        # training network: Q
        # for computing gradient of (y-Q(s,a))**2
        with tf.variable_scope('Q'):
            Q_action = q_fn(state,action)

        # training network: policy
        # for input into Q_policy below
        with tf.variable_scope('policy'):
            policy = self.policy_fn(state)

        # for computing policy gradient w.r.t. Q(state,policy)
        with tf.variable_scope('Q',reuse=True):
            Q_policy = self.q_fn(state,policy)

        # target networks
        ema_op, ema_vars_getter = exponential_moving_average(
                ['Q','policy'],decay=ema_decay,zero_debias=True)

        with tf.variable_scope('policy',reuse=True,custom_getter=ema_vars_getter):
            policy_ema = self.policy_fn(state2,training=False)

        with tf.variable_scope('Q',reuse=True,custom_getter=ema_vars_getter):
            Q_ema_state2 = self.q_fn(state2,policy_ema,training=False)

        # loss functions
        y = reward + gamma*(1-done)*Q_ema_state2
        Q_err = Q_action-tf.stop_gradient(y) #the stop gradient isn't totally necessary
        loss_Q = 0.5*tf.reduce_mean(tf.square(Q_err))
        loss_policy, _ = dpg(Q_policy,policy,self.dqda_clipping,self.clip_norm)
        #loss_policy = -tf.reduce_mean(Q_policy) #negative to maximize Q using a minimizer
        loss = loss_Q + loss_policy

        # gradient update for parameters of Q 
        opt = tf.train.RMSPropOptimizer(learning_rate) 
        train_op = opt_Q.minimize(loss)

        # used in update step
        self.update_ops = [ema_op,train_op]

        # store attributes for later use
        self.state = state
        self.action = action
        self.reward = reward
        self.state2 = state2
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.ema_decay = ema_decay
        self.ema_op = ema_op
        self.train_Q_op = train_Q_op
        self.train_policy_op = train_policy_op
        self.loss = loss
        self.loss_Q = loss_Q
        self.loss_policy = loss_policy
        self.policy = policy
        self.Q = Q
        self.policy_ema = policy_ema
        self.Q_ema_state2 = Q_ema_state2
        
    def act(self,sess,state):
        return sess.run(self.policy,{self._state:state})

    def update(self,sess,training_sample,gamma=0.99,learning_rate=1e-3,ema_decay=0.999):
        sess.run(
            update_ops,
            {
                self.state:training_sample['state'],
                self.action:training_sample['action'],
                self.reward:training_sample['reward'],
                self.state2:training_sample['state2'],
                self.gamma:gamma,
                self.learning_rate:learning_rate,
                self.ema_decay:ema_decay,
            }
        )
