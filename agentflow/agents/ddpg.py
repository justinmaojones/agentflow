import tensorflow as tf
import numpy as np
from ..objectives import dpg, td_learning
from ..tensorflow.ops import exponential_moving_average
from ..tensorflow.ops import entropy_loss
from ..tensorflow.ops import l2_loss

class DDPG(object):

    def __init__(self,state_shape,action_shape,policy_fn,q_fn,dqda_clipping=None,clip_norm=False,discrete=False):
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
          discrete: Whether to treat policy as discrete or continuous.

        """
        self.state_shape = list(state_shape)
        self.action_shape = list(action_shape)
        self.policy_fn = policy_fn
        self.q_fn = q_fn
        self.dqda_clipping = dqda_clipping
        self.clip_norm = clip_norm
        self.discrete = discrete
        self.build_model()

    def build_model(self):

        with tf.variable_scope(None,default_name='DDPG') as scope:

            # inputs
            inputs = {
                'state': tf.placeholder(tf.float32,shape=tuple([None]+self.state_shape)),
                'action': tf.placeholder(tf.float32,shape=tuple([None]+self.action_shape)),
                'reward': tf.placeholder(tf.float32,shape=(None,)),
                'done': tf.placeholder(tf.float32,shape=(None,)),
                'state2': tf.placeholder(tf.float32,shape=tuple([None]+self.state_shape)),
                'gamma': tf.placeholder(tf.float32,shape=()),
                'learning_rate': tf.placeholder(tf.float32,shape=()),
                'ema_decay': tf.placeholder(tf.float32,shape=()),
                'importance_weight': tf.placeholder(tf.float32,shape=(None,)),
                'weight_decay': tf.placeholder(tf.float32,shape=()),
                'policy_loss_weight': tf.placeholder(tf.float32,shape=()),
                'entropy_loss_weight': tf.placeholder(tf.float32,shape=(),name='entropy_loss_weight'),
            }
            self.inputs = inputs

            # build training networks

            # training network: policy
            # for input into Q_policy below
            with tf.variable_scope('policy'):
                policy_train, policy_train_logits, _ = self.policy_fn(inputs['state'],training=True)
            
            # for evaluation in the environment
            with tf.variable_scope('policy',reuse=True):
                policy_eval, _, _ = self.policy_fn(inputs['state'],training=False)

            # training network: Q
            # for computing TD (time-delay) learning loss
            with tf.variable_scope('Q'):
                Q_action_train = self.q_fn(inputs['state'],inputs['action'],training=True)

            # for computing policy gradient w.r.t. Q(state,policy)
            with tf.variable_scope('Q',reuse=True):
                Q_policy_train = self.q_fn(inputs['state'],policy_train,training=False)

            # target networks
            ema, ema_op, ema_vars_getter = exponential_moving_average(
                    scope.name,decay=inputs['ema_decay'],zero_debias=True)

            with tf.variable_scope('policy',reuse=True,custom_getter=ema_vars_getter):
                policy_ema, _, _ = self.policy_fn(inputs['state'],training=False)
                if self.discrete:
                    pe_depth = self.action_shape[-1] 
                    pe_indices = tf.argmax(policy_ema,axis=-1)
                    policy_ema = tf.one_hot(pe_indices,pe_depth)

            with tf.variable_scope('policy',reuse=True,custom_getter=ema_vars_getter):
                policy_ema_state2, _, _ = self.policy_fn(inputs['state2'],training=False)
                if self.discrete:
                    pe_depth = self.action_shape[-1] 
                    pe_indices = tf.argmax(policy_ema_state2,axis=-1)
                    policy_ema_state2 = tf.one_hot(pe_indices,pe_depth)

            with tf.variable_scope('Q',reuse=True,custom_getter=ema_vars_getter):
                Q_ema_state2 = self.q_fn(inputs['state2'],policy_ema_state2,training=False)

            # make sure inputs to loss functions are in the correct shape
            # (to avoid erroneous broadcasting)
            reward = tf.reshape(inputs['reward'],[-1])
            done = tf.reshape(inputs['done'],[-1])
            Q_action_train = tf.reshape(Q_action_train,[-1])
            Q_ema_state2 = tf.reshape(Q_ema_state2,[-1])

            # loss functions
            losses_Q, y, td_error = td_learning(
                Q_action_train,
                reward,
                inputs['gamma'],
                (1-done)*Q_ema_state2
            )
            losses_policy = dpg(Q_policy_train,policy_train,self.dqda_clipping,self.clip_norm)
            losses_entropy_reg = entropy_loss(policy_train_logits)

            assert losses_Q.shape.as_list() == [None]
            assert losses_policy.shape.as_list() == [None]
            assert losses_entropy_reg.shape.as_list() == [None]

            # overall loss function (importance weighted)
            loss = tf.reduce_mean(
                self.inputs['importance_weight']*tf.add_n([
                    losses_Q,
                    inputs['policy_loss_weight']*losses_policy,
                    inputs['entropy_loss_weight']*losses_entropy_reg,
                ])
            )

            # weight decay
            loss += inputs['weight_decay']*l2_loss(scope.name)

            # gradient update for parameters of Q 
            self.optimizer = tf.train.AdamOptimizer(inputs['learning_rate']) 
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS,scope=scope.name)):
                train_op = self.optimizer.minimize(loss)

            # used in update step
            self.update_ops = {
                'ema': ema_op,
                'train': train_op
            }

            # policy gradient
            policy_gradient = tf.gradients(losses_policy,policy_train)
            policy_gradient_norm = tf.norm(policy_gradient,ord=2,axis=1)

            # store attributes for later use
            self.outputs = {
                'loss': loss,
                'losses_Q': losses_Q,
                'losses_entropy_reg': losses_entropy_reg,
                'losses_policy': losses_policy,
                'policy_ema': policy_ema,
                'policy_ema_state2': policy_ema_state2,
                'policy_eval': policy_eval,
                'policy_gradient': policy_gradient,
                'policy_gradient_norm': policy_gradient_norm,
                'policy_train': policy_train,
                'Q_action_train': Q_action_train,
                'Q_ema_state2': Q_ema_state2,
                'Q_policy_train': Q_policy_train,
                'td_error': td_error,
                'y': y,
            }
        
    def act(self,state,session=None):
        session = session or tf.get_default_session()
        return session.run(self.outputs['policy_eval'],{self.inputs['state']:state})
        
    def act_train(self,state,session=None):
        session = session or tf.get_default_session()
        return session.run(self.outputs['policy_train'],{self.inputs['state']:state})

    def get_inputs(self,**inputs):
        return {self.inputs[k]: inputs[k] for k in inputs}

    def update(self,state,action,reward,done,state2,gamma=0.99,learning_rate=1e-3,ema_decay=0.999,weight_decay=0.1,importance_weight=None,entropy_loss_weight=1.0,policy_loss_weight=1.0,session=None,outputs=['td_error']):
        session = session or tf.get_default_session()
        if importance_weight is None:
            importance_weight = np.ones_like(reward)
        inputs = {
            self.inputs['state']:state,
            self.inputs['action']:action,
            self.inputs['reward']:reward,
            self.inputs['done']:done,
            self.inputs['state2']:state2,
            self.inputs['gamma']:gamma,
            self.inputs['learning_rate']:learning_rate,
            self.inputs['ema_decay']:ema_decay,
            self.inputs['weight_decay']:weight_decay,
            self.inputs['importance_weight']:importance_weight,
            self.inputs['policy_loss_weight']:policy_loss_weight,
            self.inputs['entropy_loss_weight']:entropy_loss_weight,
        }

        my_outputs, _ = session.run(
            [
                {k:self.outputs[k] for k in outputs},
                self.update_ops
            ],
            inputs
        )
        return my_outputs
