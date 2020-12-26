import numpy as np
import tensorflow as tf
from tensorflow.python.framework.errors_impl import InvalidArgumentError

from ..objectives import td_learning
from ..tensorflow.ops import exponential_moving_average
from ..tensorflow.ops import entropy_loss
from ..tensorflow.ops import l2_loss

class DQN(object):

    def __init__(self,state_shape,num_actions,q_fn):
        """Implements Deep Deterministic Policy Gradient with Tensorflow

        This class builds a DDPG model with optimization update and action prediction steps.

        Args:
          state_shape: a tuple or list of the state shape, excluding the batch dimension.
            For example, for images of size 28 x 28 x 3, state_shape=[28,28,3].
          num_actions: an integer, representing the number of possible actions an agent can choose
            from, excluding the batch dimension. It is assumed that actions are one-hot, 
            i.e. "one of `num_actions`".
          q_fn: a function that takes as input two tensors: the state and action,
            and outputs an estimate Q(state,action)
        """
        self.state_shape = list(state_shape)
        self.num_actions = num_actions 
        self.q_fn = q_fn

        self.build_model()

    def build_model(self):

        with tf.variable_scope(None,default_name='DQN') as scope:

            # inputs
            inputs = {
                'state': tf.placeholder(tf.float32,shape=tuple([None]+self.state_shape)),
                'action': tf.placeholder(tf.float32,shape=[None, self.num_actions]),
                'reward': tf.placeholder(tf.float32,shape=(None,)),
                'returns': tf.placeholder(tf.float32,shape=(None,)),
                'done': tf.placeholder(tf.float32,shape=(None,)),
                'state2': tf.placeholder(tf.float32,shape=tuple([None]+self.state_shape)),
                'gamma': tf.placeholder(tf.float32),
                'learning_rate': tf.placeholder(tf.float32),
                'ema_decay': tf.placeholder(tf.float32),
                'importance_weight': tf.placeholder(tf.float32,shape=(None,)),
                'weight_decay': tf.placeholder(tf.float32,shape=()),
                'entropy_loss_weight': tf.placeholder(tf.float32,shape=(),name='entropy_loss_weight'),
            }
            self.inputs = inputs

            # training network: Q
            # for computing TD (time-delay) learning loss
            with tf.variable_scope('Q'):
                Q_train = self.q_fn(inputs['state'],training=True)

                if not (Q_train.shape.as_list() == inputs['action'].shape.as_list()):
                    raise InvalidArgumentError("Q_train shape (%s) and action shape (%s) must match" % \
                                                 (str(Q_train.shape), str(inputs['action'].shape)))

                Q_action_train = tf.reduce_sum(Q_train*inputs['action'],axis=-1,keepdims=True)
                policy_train = tf.nn.softmax(Q_train,axis=-1)

            with tf.variable_scope('Q',reuse=True):
                Q_eval = self.q_fn(inputs['state'],training=False)

                if not (Q_eval.shape.as_list() == inputs['action'].shape.as_list()):
                    raise InvalidArgumentError("Q_eval shape (%s) and action shape (%s) must match" % \
                                                 (str(Q_eval.shape), str(inputs['action'].shape)))

                Q_action_eval = tf.reduce_sum(Q_train*inputs['action'],axis=-1,keepdims=True)
                policy_eval = tf.nn.softmax(Q_eval,axis=-1)
                pe_indices = tf.argmax(policy_eval, axis=-1)
                Q_policy_eval = tf.one_hot(pe_indices, self.num_actions)

            # target networks
            ema, ema_op, ema_vars_getter = exponential_moving_average(
                    scope.name,decay=inputs['ema_decay'],zero_debias=True)

            with tf.variable_scope('Q',reuse=True,custom_getter=ema_vars_getter):
                Q_ema_state2 = self.q_fn(inputs['state'],training=False)
                Q_ema_state2 = tf.reduce_max(Q_ema_state2,axis=-1)

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

            assert losses_Q.shape == tf.TensorShape(None)

            # entropy regularization
            losses_entropy_reg = tf.squeeze(entropy_loss(Q_train))
            assert losses_entropy_reg.shape == tf.TensorShape(None)

            #loss = tf.reduce_mean(self.inputs['importance_weight']*losses_Q)
            # overall loss function (importance weighted)
            loss = tf.reduce_mean(
                self.inputs['importance_weight']*tf.add_n([
                    losses_Q,
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

            # store attributes for later use
            self.outputs = {
                'loss': loss,
                'losses_Q': losses_Q,
                'policy_train': policy_train,
                'policy_eval': policy_eval,
                'Q_action_eval': Q_action_eval,
                'Q_action_train': Q_action_train,
                'Q_policy_eval': Q_policy_eval,
                'Q_ema_state2': Q_ema_state2,
                'td_error': td_error,
                'y': y,
            }

    def act(self,state,session=None):
        return self.act_probs(state,session).argmax(axis=-1).ravel()
        
    def act_probs(self,state,session=None):
        session = session or tf.get_default_session()
        return session.run(self.outputs['policy_eval'],{self.inputs['state']:state})
        
    def get_inputs(self,**inputs):
        return {self.inputs[k]: inputs[k] for k in inputs}

    def infer(self,outputs=None,session=None,**inputs):
        session = session or tf.get_default_session()
        inputs = self.get_inputs(**inputs)
        if outputs is None:
            outputs = self.outputs
        else:
            outputs = {k:self.outputs[k] for k in outputs}
        return session.run(outputs,inputs)

    def update(self,state,action,reward,done,state2,gamma=0.99,learning_rate=1.,ema_decay=0.999,weight_decay=0.1,importance_weight=None,entropy_loss_weight=0.0,session=None,outputs=['td_error'],returns=None):
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
