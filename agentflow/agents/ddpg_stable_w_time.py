import tensorflow as tf
import numpy as np
from ..objectives import dpg, td_learning
from ..tensorflow.ops import exponential_moving_average 
from .ddpg_stable import get_modified_gradients_pinv


class StableDDPGTime(object):

    def __init__(self,state_shape,action_shape,policy_fn,q_fn,dqda_clipping=None,clip_norm=False,discrete=False,episodic=True,beta=1,alpha=1):
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
          episodic: W.

        """
        self.state_shape = list(state_shape)
        self.action_shape = list(action_shape)
        self.policy_fn = policy_fn
        self.q_fn = q_fn
        self.dqda_clipping = dqda_clipping
        self.clip_norm = clip_norm
        self.discrete = discrete
        self.episodic = episodic
        self.alpha = alpha
        self.beta = beta
        self.build_model()

    def build_model(self):

        with tf.variable_scope(None,default_name='DDPG') as scope:

            # inputs
            inputs = {
                'state': tf.placeholder(tf.float32,shape=tuple([None]+self.state_shape)),
                'time': tf.placeholder(tf.float32,shape=(None,)),
                'action': tf.placeholder(tf.float32,shape=tuple([None]+self.action_shape)),
                'reward': tf.placeholder(tf.float32,shape=(None,)),
                'done': tf.placeholder(tf.float32,shape=(None,)),
                'state2': tf.placeholder(tf.float32,shape=tuple([None]+self.state_shape)),
                'time2': tf.placeholder(tf.float32,shape=(None,)),
                'gamma': tf.placeholder(tf.float32),
                'learning_rate': tf.placeholder(tf.float32),
                'learning_rate_q': tf.placeholder(tf.float32),
                'ema_decay': tf.placeholder(tf.float32),
                'importance_weight': tf.placeholder(tf.float32,shape=(None,)),
                'weight_decay': tf.placeholder(tf.float32,shape=()),
                'learning_rate': tf.placeholder(tf.float32),
            }
            self.inputs = inputs

            # build training networks

            # training network: policy
            # for input into Q_policy below
            with tf.variable_scope('policy'):
                policy_train = self.policy_fn(inputs['state'],inputs['time'],training=True)
            
            # for evaluation in the environment
            with tf.variable_scope('policy',reuse=True):
                policy_eval = self.policy_fn(inputs['state'],inputs['time'],training=False)

            # training network: Q
            # for computing TD (time-delay) learning loss
            with tf.variable_scope('Q'):
                Q_action_train = self.q_fn(inputs['state'],inputs['action'],inputs['time'],training=True)

            # training network: Reward
            with tf.variable_scope('R'):
                R_action_train = self.q_fn(inputs['state'],inputs['action'],inputs['time'],training=True)

            # for computing policy gradient w.r.t. Q(state,policy)
            with tf.variable_scope('Q',reuse=True):
                Q_policy_train = self.q_fn(inputs['state'],policy_train,inputs['time'],training=False)

            # target networks
            ema, ema_op, ema_vars_getter = exponential_moving_average(
                    scope.name,decay=inputs['ema_decay'],zero_debias=True)

            with tf.variable_scope('policy',reuse=True,custom_getter=ema_vars_getter):
                policy_ema_probs = self.policy_fn(inputs['state'],inputs['time'],training=False)
                if self.discrete:
                    pe_depth = self.action_shape[-1] 
                    pe_indices = tf.argmax(policy_ema_probs,axis=-1)
                    policy_ema = tf.one_hot(pe_indices,pe_depth)

            with tf.variable_scope('policy',reuse=True,custom_getter=ema_vars_getter):
                policy_ema_state2_probs = self.policy_fn(inputs['state2'],inputs['time2'],training=False)
                if self.discrete:
                    pe_depth = self.action_shape[-1] 
                    pe_indices = tf.argmax(policy_ema_state2_probs,axis=-1)
                    policy_ema_state2 = tf.one_hot(pe_indices,pe_depth)

            #with tf.variable_scope('Q',reuse=True,custom_getter=ema_vars_getter):
            with tf.variable_scope('Q',reuse=True):
                Q_ema_state2 = self.q_fn(inputs['state2'],policy_ema_state2,inputs['time2'],training=False)

            with tf.variable_scope('R',reuse=True,custom_getter=ema_vars_getter):
                R_ema = self.q_fn(inputs['state'],policy_ema,inputs['time'],training=False)

            # make sure inputs to loss functions are in the correct shape
            # (to avoid erroneous broadcasting)
            reward = tf.reshape(inputs['reward'],[-1])
            done = tf.reshape(inputs['done'],[-1])
            Q_action_train = tf.reshape(Q_action_train,[-1])
            Q_ema_state2 = tf.reshape(Q_ema_state2,[-1])
            R_action_train = tf.reshape(R_action_train,[-1])
            R_ema = tf.reshape(R_ema,[-1])

            # average reward
            reward_avg = tf.Variable(tf.zeros(1),dtype=tf.float32,name='avg_reward')

            # loss functions
            if self.episodic:
                losses_Q, y, td_error = td_learning(
                        Q_action_train,reward,inputs['gamma'],(1-done)*Q_ema_state2)
                loss_R = 1.
            else:
                reward_differential = tf.stop_gradient(reward) - reward_avg 
#                losses_Q, y, td_error = td_learning(
#                        Q_action_train,reward_differential,inputs['gamma'],Q_ema_state2)
                if False:
                    losses_Q, y, td_error = td_learning(
                            Q_action_train,
                            (1-inputs['gamma'])*tf.stop_gradient(reward) - reward_avg,
                            inputs['gamma'],
                            Q_ema_state2)
                else:
                    losses_Q, y, td_error = td_learning(
                            Q_action_train,(1-inputs['gamma'])*reward,inputs['gamma'],Q_ema_state2)

                #loss_R = 0.5*tf.square(R_action_train - reward)
                loss_R = 0.5*tf.square(
                        reward_differential+tf.stop_gradient(Q_ema_state2-Q_action_train))
            losses_policy = dpg(Q_policy_train,policy_train,self.dqda_clipping,self.clip_norm)
            loss_policy = tf.reduce_mean(self.inputs['importance_weight']*losses_policy)
            loss = tf.reduce_mean(self.inputs['importance_weight']*(losses_Q + losses_policy + loss_R))

            # policy gradient
            policy_gradient = tf.gradients(losses_policy,policy_train)

            # weight decay
            variables = []
            for v in tf.trainable_variables(scope=scope.name):
                if v != reward_avg:
                    variables.append(v)
            l2_reg = inputs['weight_decay']*tf.reduce_sum([tf.nn.l2_loss(v) for v in variables])
            loss += l2_reg

            # stable gradients for parameters of Q
            with tf.variable_scope('Q') as scope_Q:
                self.var_list_Q = tf.trainable_variables(scope=scope_Q.name)
                grad_Q, _ = get_modified_gradients_pinv(
                    self.var_list_Q,
                    Q_action_train,
                    Q_ema_state2,
                    td_error,
                    alpha=self.alpha,
                    beta=self.beta,
                )
            self.optimizer_Q = tf.train.GradientDescentOptimizer(inputs['learning_rate_q'])
            train_op_Q = self.optimizer_Q.apply_gradients(grad_Q)
            
            # gradient update for parameters of policy
            with tf.variable_scope('policy') as scope_policy:
                self.var_list_policy = tf.trainable_variables(scope=scope_policy.name)
            self.optimizer = tf.train.RMSPropOptimizer(inputs['learning_rate']) 
            other_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS,scope=scope.name)
            with tf.control_dependencies(other_update_ops):
                train_op = self.optimizer.minimize(loss,var_list=self.var_list_policy)

            # used in update step
            self.update_ops = {
                'ema': ema_op,
                'train': train_op,
                'train_Q': train_op_Q,
                'other_update_ops': other_update_ops,
            }

            # store attributes for later use
            self.outputs = {
                'y': y,
                'td_error': td_error,
                'loss': loss,
                'losses_Q': losses_Q,
                'losses_policy': losses_policy,
                'policy_train': policy_train,
                'policy_eval': policy_eval,
                'Q_action_train': Q_action_train,
                'Q_policy_train': Q_policy_train,
                'policy_ema_probs': policy_ema_probs,
                'policy_ema': policy_ema,
                'policy_ema_state2_probs': policy_ema_state2_probs,
                'policy_ema_state2': policy_ema_state2,
                'Q_ema_state2': Q_ema_state2,
                'R_action_train': R_action_train,
                'R_ema': R_ema,
                'reward_avg': reward_avg,
                'policy_gradient': policy_gradient,
            }

            if not self.episodic:
                self.outputs['reward_differential'] = reward_differential
        
    def act(self,state,time,session=None):
        session = session or tf.get_default_session()
        return session.run(
            self.outputs['policy_eval'],
            {
                self.inputs['state']:state.astype('float32'),
                self.inputs['time']:time.astype('float32')
            }
        )
        
    def act_train(self,state,session=None):
        session = session or tf.get_default_session()
        return session.run(self.outputs['policy_train'],{self.inputs['state']:state.astype('float32')})

    def get_inputs(self,**inputs):
        return {self.inputs[k]: inputs[k] for k in inputs}

    def update(self,state,time,action,reward,done,state2,time2,gamma=0.99,learning_rate=1e-3,learning_rate_q=1.,ema_decay=0.999,weight_decay=0.1,importance_weight=None,session=None,outputs=['td_error']):
        session = session or tf.get_default_session()
        if importance_weight is None:
            importance_weight = np.ones_like(reward)
        my_outputs, _ = session.run(
            [[self.outputs[k] for k in outputs],self.update_ops],
            {
                self.inputs['state']:state.astype('float32'),
                self.inputs['time']:time.astype('float32'),
                self.inputs['action']:action,
                self.inputs['reward']:reward,
                self.inputs['done']:done,
                self.inputs['state2']:state2.astype('float32'),
                self.inputs['time2']:time2.astype('float32'),
                self.inputs['gamma']:gamma,
                self.inputs['learning_rate']:learning_rate,
                self.inputs['learning_rate_q']:learning_rate_q,
                self.inputs['ema_decay']:ema_decay,
                self.inputs['weight_decay']:weight_decay,
                self.inputs['importance_weight']:importance_weight,
            }
        )
        return my_outputs