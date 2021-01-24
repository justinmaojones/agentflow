import numpy as np
import tensorflow as tf
from tensorflow.python.framework.errors_impl import InvalidArgumentError

from ..objectives import td_learning
from ..tensorflow.ops import exponential_moving_average
from ..tensorflow.ops import entropy_loss
from ..tensorflow.ops import l2_loss
from ..tensorflow.ops import not_trainable_getter 
from ..tensorflow.ops import onehot_argmax
from ..tensorflow.ops import value_at_argmax 

class BootstrappedDQN(object):

    def __init__(self,state_shape,num_actions,num_heads,q_fn,q_prior_fn=None,double_q=False,random_prior=False,prior_scale=1.0):
        """Implements Boostrapped Deep Q Networks [1] with Tensorflow

        This class builds a DDPG model with optimization update and action prediction steps.

        Args:
          state_shape: a tuple or list of the state shape, excluding the batch dimension.
            For example, for images of size 28 x 28 x 3, state_shape=[28,28,3].
          num_actions: an integer, representing the number of possible actions an agent can choose
            from, excluding the batch dimension. It is assumed that actions are one-hot, 
            i.e. "one of `num_actions`".
          num_heads: an integer, representing the number of Q functions used for bootstrapping.
          q_fn: a function that takes as input two tensors: the state and action,
            and outputs an estimate Q(state,action)
          q_prior_fn: a function that takes as input two tensors: the state and action,
            and outputs an estimate Q(state,action)
          double_q: boolean, when true uses "double q-learning" from [2]. Otherwise uses
            standard q-learning.
          random_prior: boolean, when true builds a separate randomized prior function [3] 
            for each bootstrap head using q_fn.
            
        References:
        [1] Osband, Ian, et al. "Deep exploration via bootstrapped DQN." Advances in neural 
            information processing systems. 2016.
        [2] Van Hasselt, Hado, Arthur Guez, and David Silver. "Deep reinforcement learning 
            with double q-learning." arXiv preprint arXiv:1509.06461 (2015).
        [3] Osband, Ian, John Aslanides, and Albin Cassirer. "Randomized prior functions for 
            deep reinforcement learning." Advances in Neural Information Processing Systems. 2018.
        """
        if q_prior_fn is not None:
            assert random_prior, "random_prior must be true if q_prior_fn is provided"

        self.state_shape = list(state_shape)
        self.num_actions = num_actions 
        self.num_heads = num_heads
        self.q_fn = q_fn
        self.q_prior_fn = q_prior_fn if q_prior_fn is not None else q_fn
        self.double_q = double_q
        self.random_prior = random_prior
        self.prior_scale = prior_scale

        self.build_model()

    def _validate_q_fn(self, Q):
        if not (Q.shape.as_list() == [None, self.num_actions, self.num_heads]):
            err_msg = "expected q_fn to output tensor with shape [None, %d, %d], but instead got %s" % \
                        (self.num_actions, self.num_heads, str(Q.shape.as_list()))
            raise ValueError(err_msg)

    def build_model(self):

        with tf.variable_scope(None,default_name='BootstrappedDQN') as scope:

            # inputs
            inputs = {
                'state': tf.placeholder(tf.float32,shape=tuple([None]+self.state_shape), name='state'),
                'action': tf.placeholder(tf.float32,shape=[None, self.num_actions], name='action'),
                'reward': tf.placeholder(tf.float32,shape=(None,), name='reward'),
                'done': tf.placeholder(tf.float32,shape=(None,), name='done'),
                'state2': tf.placeholder(tf.float32,shape=tuple([None]+self.state_shape), name='state2'),
                'gamma': tf.placeholder(tf.float32, shape=(), name='gamma'),
                'learning_rate': tf.placeholder(tf.float32, shape=(), name='learning_rate'),
                'ema_decay': tf.placeholder(tf.float32, shape=(), name='ema_decay'),
                'importance_weight': tf.placeholder(tf.float32,shape=(None,), name='importance_weight'),
                'weight_decay': tf.placeholder(tf.float32,shape=(), name='weight_decay'),
                'entropy_loss_weight': tf.placeholder(tf.float32,shape=(),name='entropy_loss_weight'),
                'mask': tf.placeholder(tf.float32, shape=(None, self.num_heads), name='mask')
            }
            self.inputs = inputs

            def weighted_avg(x, weights):
                return tf.reduce_sum(x*weights,axis=-1) / tf.reduce_sum(weights,axis=-1)

            if self.random_prior:
                def q_fn(state, training):
                    # since prior variables are not trainable, the exponential moving average
                    # getter will not process prior variables, but instead just return the
                    # prior variables themselves
                    with tf.variable_scope('Q_prior',custom_getter=not_trainable_getter):
                        Q_prior_multihead = tf.stop_gradient(self.q_prior_fn(state,training))
                        self._validate_q_fn(Q_prior_multihead)
                    Q_multihead = self.q_fn(state, training) 
                    self._validate_q_fn(Q_multihead)
                    return Q_multihead + self.prior_scale * Q_prior_multihead

            else:
                def q_fn(state, training):
                    Q_multihead = self.q_fn(state, training) 
                    self._validate_q_fn(Q_multihead)
                    return Q_multihead

            # training network: Q
            # for computing TD (time-delay) learning loss
            with tf.variable_scope('Q'):
                # the bootstrapped heads should be in the last dimension
                # Q_train_multihead is shape (None, num_actions, num_heads)
                Q_train_multihead = q_fn(inputs['state'],training=True)

                # Q_train is shape (None, num_actions)
                Q_train = weighted_avg(Q_train_multihead, inputs['mask'][:,None,:]) 

                # Q_action_train_multihead is shape (None, num_heads) and represents Q(s,a) for each head
                Q_action_train_multihead = tf.reduce_sum(Q_train_multihead*inputs['action'][:,:,None],axis=-2)

                # Q_action_train is shape (None,) and represents the masked average
                Q_action_train = tf.reduce_sum(Q_train*inputs['action'],axis=-1)
                policy_train = tf.nn.softmax(Q_train,axis=-1)

            with tf.variable_scope('Q',reuse=True):
                # Q_eval_multihead is shape (None, num_actions, num_heads)
                Q_eval_multihead = q_fn(inputs['state'],training=False)

                # Q_eval is shape (None, num_actions)
                Q_eval = weighted_avg(Q_eval_multihead, inputs['mask'][:,None,:])

                # Q_action_eval is shape (None,) and represents the masked average
                Q_action_eval = tf.reduce_sum(Q_eval*inputs['action'],axis=-1)
                Q_policy_eval = tf.reduce_max(Q_eval,axis=-1)
                policy_eval = tf.nn.softmax(Q_eval,axis=-1)

            with tf.variable_scope('Q',reuse=True):
                # Q_state2_eval_multihead is shape (None, num_actions, num_heads)
                Q_state2_eval_multihead = q_fn(inputs['state2'],training=False)
                # Q_state2_eval is shape (None, num_actions)
                Q_state2_eval = weighted_avg(Q_state2_eval_multihead, inputs['mask'][:,None,:])

            # target networks
            ema, ema_op, ema_vars_getter = exponential_moving_average(
                    scope.name,decay=inputs['ema_decay'],zero_debias=True)

            with tf.variable_scope('Q',reuse=True,custom_getter=ema_vars_getter):
                # Q_ema_state2_multihead is shape (None, num_actions, num_heads)
                Q_ema_state2_multihead = q_fn(inputs['state2'],training=False)
                # Q_ema_state2 is shape (None, num_actions)
                Q_ema_state2 = weighted_avg(Q_ema_state2_multihead, inputs['mask'][:,None,:])

            # target_Q_state2 is shape (None, num_heads)
            if self.double_q:
                # use action = argmax(Q_state2_eval_multihead, axis=-2) to select Q 
                # from axis 2 of Q_ema_state2_multihead
                target_Q_state2 = value_at_argmax(Q_state2_eval_multihead, Q_ema_state2_multihead, axis=-2) 
            else: 
                target_Q_state2 = tf.reduce_max(Q_ema_state2_multihead,axis=-2)

            # loss functions
            losses_Q_multihead, y, td_error_multihead = td_learning(
                Q_action_train_multihead,
                inputs['reward'][:,None],
                inputs['gamma'],
                (1-inputs['done'][:,None])*target_Q_state2
            )

            td_error = weighted_avg(td_error_multihead, inputs['mask']) 
            abs_td_error = weighted_avg(tf.abs(td_error_multihead), inputs['mask']) 

            losses_Q = tf.reduce_sum(losses_Q_multihead*inputs['mask'], axis=-1)
            losses_Q /= self.num_heads # gradient normalization 
            assert losses_Q.shape.as_list() == [None]

            # entropy regularization
            losses_entropy_reg_multihead = entropy_loss(Q_train_multihead, axis=-2)
            losses_entropy_reg = tf.reduce_sum(losses_entropy_reg_multihead*inputs['mask'], axis=-1)
            assert losses_entropy_reg.shape.as_list() == [None]

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


            self.var_list = tf.global_variables(scope=scope.name)
            self._pnorms = {v.name: tf.sqrt(tf.reduce_mean(tf.square(v))) for v in self.var_list} 

            # store attributes for later use
            self.outputs = {
                'loss': loss,
                'losses_Q': losses_Q,
                'policy_train': policy_train,
                'policy_eval': policy_eval,
                'Q_action_eval': Q_action_eval,
                'Q_action_train': Q_action_train,
                'Q_policy_eval': Q_policy_eval,
                'Q_state2_eval': Q_state2_eval,
                'Q_ema_state2': Q_ema_state2,
                'target_Q_state2': target_Q_state2,
                'abs_td_error': abs_td_error,
                'td_error': td_error,
                'y': y,
            }

    def act(self,state,mask=None,session=None,addl_outputs=None):
        if session is None:
            session = tf.get_default_session()
        if mask is None:
            mask = np.ones((len(state),self.num_heads))
        if addl_outputs is None:
            addl_outputs = []
        assert isinstance(addl_outputs, list)

        outputs = ['policy_eval'] + addl_outputs
        output = session.run(
            {k: self.outputs[k] for k in outputs},
            {
                self.inputs['state']:state,
                self.inputs['mask']:mask,
            }
        )
        output['action'] = output['policy_eval'].argmax(axis=-1).ravel()
        return output
        
    def act_probs(self,state,mask=None,session=None):
        if mask is None:
            mask = np.ones((len(state),self.num_heads))
        session = session or tf.get_default_session()
        output = session.run(
            self.outputs['policy_eval'],
            {
                self.inputs['state']:state,
                self.inputs['mask']:mask,
            }
        )
        return output
        
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

    def pnorms(self,session=None):
        session = session or tf.get_default_session()
        return session.run(self._pnorms)

    def update(self,state,action,reward,done,state2,mask,gamma=0.99,learning_rate=1.,ema_decay=0.999,weight_decay=0.1,importance_weight=None,entropy_loss_weight=0.0,session=None,outputs=['td_error']):
        session = session or tf.get_default_session()
        if importance_weight is None:
            importance_weight = np.ones_like(reward)
        inputs = {
            self.inputs['state']:state,
            self.inputs['action']:action,
            self.inputs['reward']:reward,
            self.inputs['done']:done,
            self.inputs['state2']:state2,
            self.inputs['mask']:mask,
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
