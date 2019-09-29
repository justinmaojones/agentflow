import tensorflow as tf
from ..objectives.trfl import dpg, td_learning
from ..tensorflow.ops import exponential_moving_average

_arg_scope = tf.contrib.framework.arg_scope
_arg_scope_fns = [tf.layers.batch_normalization,tf.layers.dropout]

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
          dqda_clipping: `int` or `float`, clips the gradient dqda element-wise
            between `[-dqda_clipping, dqda_clipping]`.
          clip_norm: Whether to perform dqda clipping on the vector norm of the last
            dimension, or component wise (default).

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
        inputs = {
            'state': tf.placeholder(tf.float32,shape=tuple([None]+self.state_shape)),
            'action': tf.placeholder(tf.int32,shape=tuple([None]+self.action_shape)),
            'reward': tf.placeholder(tf.float32,shape=(None,)),
            'done': tf.placeholder(tf.float32,shape=(None,)),
            'state2': tf.placeholder(tf.float32,shape=tuple([None]+self.state_shape)),
            'gamma': tf.placeholder(tf.float32),
            'learning_rate': tf.placeholder(tf.float32),
            'ema_decay': tf.placeholder(tf.float32),
        }
        self.inputs = inputs

        # build training networks
        with _argscope(_argscope_fns,training=True):

            # training network: policy
            # for input into Q_policy below
            with tf.variable_scope('policy'):
                policy_train = self.policy_fn(inputs['state'])

            # training network: Q
            # for computing gradient of (y-Q(s,a))**2
            with tf.variable_scope('Q'):
                Q_action_train = q_fn(inputs['state'],inputs['action'])

            # for computing policy gradient w.r.t. Q(state,policy)
            with tf.variable_scope('Q',reuse=True):
                Q_policy_train = self.q_fn(inputs['state'],policy_train)

        # target networks
        ema_op, ema_vars_getter = exponential_moving_average(
                ['Q','policy'],decay=inputs['ema_decay'],zero_debias=True)

        with _argscope(_argscope_fns,training=False):
            with tf.variable_scope('policy',reuse=True,custom_getter=ema_vars_getter):
                policy_ema = self.policy_fn(inputs['state2'],training=False)

            with tf.variable_scope('Q',reuse=True,custom_getter=ema_vars_getter):
                Q_ema_state2 = self.q_fn(inputs['state2'],policy_ema,training=False)

        # loss functions
        loss_Q = td_learning(Q_action,inputs['reward'],inputs['gamma'],Q_ema_state2)
        loss_policy = dpg(Q_policy,policy,self.dqda_clipping,self.clip_norm)
        loss = loss_Q + loss_policy

        # gradient update for parameters of Q 
        self.optimizer = tf.train.RMSPropOptimizer(inputs['learning_rate']) 
        train_op = self.optimizer.minimize(loss)

        # used in update step
        self.update_ops = {
            'ema': ema_op,
            'train': train_op
        }

        # store attributes for later use
        self.outputs = {
            'loss': loss,
            'loss_Q': loss_Q,
            'loss_policy': loss_policy,
            'policy': policy,
            'Q': Q,
            'policy_ema': policy_ema,
            'Q_ema_state2': Q_ema_state2,
        }
        
    def act(self,state,session=None):
        session = session or tf.get_default_session()
        return session.run(self.outputs['policy'],{self.inputs['state']:state})

    def update(self,state,action,reward,state2,gamma=0.99,learning_rate=1e-3,ema_decay=0.999,session=None):
        session = session or tf.get_default_session()
        sess.run(
            update_ops,
            {
                self.inputs['state']:state,
                self.inputs['action']:action,
                self.inputs['reward']:reward,
                self.inputs['state2']:state2,
                self.inputs['gamma']:gamma,
                self.inputs['learning_rate']:learning_rate,
                self.inputs['ema_decay']:ema_decay,
            }
        )
