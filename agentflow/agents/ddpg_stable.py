import tensorflow as tf
import numpy as np
from ..objectives import dpg, td_learning
from ..tensorflow.ops import exponential_moving_average, get_gradient_matrix, get_connected_vars, entropy_loss
from tensorflow.python.ops import gen_linalg_ops

def ste_prob_to_action(probs,axis=1):
    """
    straight through gradient estimator
    """
    shape = tf.shape(probs)
    depth = shape[axis]
    argmax = tf.argmax(probs,axis)
    onehot = tf.one_hot(argmax,depth)
    return tf.stop_gradient(onehot-probs)+probs

def online_lstsq(A,b,l2_regularizer,decay=0.99):
    n = A.shape[1].value
    t = tf.Variable(tf.ones(1,dtype=tf.float32),trainable=False,name="online_lstsq_t")
    AA_running = tf.Variable(tf.zeros((n,n),dtype=tf.float32),trainable=False,name="online_lstsq_AA_running")
    Ab_running = tf.Variable(tf.zeros((n,1),dtype=tf.float32),trainable=False,name="online_lstsq_Ab_running")
    AA = decay*AA_running + (1-decay)*tf.matmul(A,A,adjoint_a=True)
    Ab = decay*Ab_running + (1-decay)*tf.matmul(A,b,adjoint_a=True)
    correction = (1-decay**t)
    AA_corrected = AA/correction
    Ab_corrected = Ab/correction
    identity = tf.eye(n,dtype=tf.float32)
    regularized_AA_corrected = AA_corrected + l2_regularizer*identity
    chol = gen_linalg_ops.cholesky(regularized_AA_corrected)
    output = tf.cholesky_solve(chol, Ab_corrected) 
    update_ops = [
        tf.assign(t,t+1,name="update_online_lstsq_t"),
        tf.assign(AA_running,AA,name="update_online_lstsq_AA_running"),
        tf.assign(Ab_running,Ab,name="update_online_lstsq_Ab_running"),
    ]
    for op in update_ops:
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS,op)
    return output

def online_lstsq2(A,b,l2_regularizer):
    n = A.shape[1].value
    vprev = tf.Variable(tf.zeros((n,1),dtype=tf.float32),trainable=False,name="vprev")
    mu = tf.matmul(A,vprev)
    b2 = b - mu
    x = tf.linalg.lstsq(A,b2,fast=False,l2_regularizer=l2_regularizer)
    v = x + vprev
    update_ops = [
        tf.assign(vprev,v,name="update_vprev"),
    ]
    for op in update_ops:
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS,op)
    return v 

def get_modified_gradients_pinv(var_list,y_pred,y2_pred,td_err,alpha,beta,vprev=None,fast=True,weight_decay=None,normalize_gradients=False,online=False,online_decay=0.99,online2=False):
    var_list, gradients = get_gradient_matrix(var_list,y_pred)
    var_list2, gradients2 = get_gradient_matrix(var_list,y2_pred)

    for v,v2 in zip(var_list,var_list2):
        assert v == v2

    A = gradients
    b = td_err
    if beta > 0:
        A = tf.concat([A,gradients2*(beta**0.5)],axis=0)
        zeros = tf.zeros(tf.shape(b)[0],tf.float32)
        b = tf.concat([b,zeros],axis=0)
    if vprev is not None:
        A = tf.concat([A,vprev[None]],axis=0)
        zeros = tf.zeros(1,tf.float32)
        b = tf.concat([b,zeros],axis=0)

    if normalize_gradients:
        A_norm = tf.sqrt(tf.reduce_max(tf.reduce_sum(tf.square(A),axis=1)))
        A = A/A_norm
        b = b/A_norm

    if online:
        modified_grad_flat = online_lstsq(A,b[:,None],alpha,online_decay)
    elif online2:
        modified_grad_flat = online_lstsq2(A,b[:,None],alpha)
    else:
        modified_grad_flat = tf.linalg.lstsq(A,b[:,None],fast=fast,l2_regularizer=alpha)
    #modified_grad_flat = modified_grad_flat/tf.cast(tf.shape(grads)[0],tf.float32)

    modified_grad = []
    i = 0
    for v in var_list:
        w = np.prod(v.shape).value
        g = tf.reshape(modified_grad_flat[i:i+w],v.shape)
        if weight_decay is not None:
            g += weight_decay*v
        modified_grad.append((g,v))
        i += w

    supplementary_output = {
        'gradients': gradients,
        'gradients2': gradients2,
        'modified_grad_flat': modified_grad_flat,
    }
    return modified_grad, supplementary_output

def l2_loss(t_list,weight_decay):
    return weight_decay*tf.add_n(list(map(tf.nn.l2_loss,t_list)))

class StableDDPG(object):

    def __init__(self,state_shape,action_shape,policy_fn,q_fn,dqda_clipping=None,
            clip_norm=False,discrete=False,episodic=True,beta=1,alpha=1,
            optimizer_q='gradient_descent',opt_q_layerwise=False,optimizer_q_kwargs=None,
            regularize_policy=True,straight_through_estimation=False,
            add_return_loss=False,stable=True,
            opt_stable_q_online=False,opt_stable_q_online_momentum=0.99,
        ):
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
        self.optimizer_q = optimizer_q
        self.opt_q_layerwise = opt_q_layerwise
        self.optimizer_q_kwargs = optimizer_q_kwargs
        self.regularize_policy = regularize_policy
        self.straight_through_estimation = straight_through_estimation
        self.add_return_loss = add_return_loss
        self.stable = stable
        self.opt_stable_q_online = opt_stable_q_online
        self.opt_stable_q_online_momentum = opt_stable_q_online_momentum

        self.build_model()

    def build_placeholder(self,tf_type,shape):
        if isinstance(shape,dict):
            return {k: self.build_placeholder(tf_type,shape[k])}
        else:
            return tf.placeholder(tf_type,shape=tuple([None]+shape))


    def build_model(self):

        with tf.variable_scope(None,default_name='DDPG') as scope:

            # inputs
            inputs = {
                'state': self.build_placeholder(tf.float32, self.state_shape),
                'action': tf.placeholder(tf.float32,shape=tuple([None]+self.action_shape)),
                'reward': tf.placeholder(tf.float32,shape=(None,)),
                'returns': tf.placeholder(tf.float32,shape=(None,)),
                'done': tf.placeholder(tf.float32,shape=(None,)),
                'state2': self.build_placeholder(tf.float32, self.state_shape),
                'gamma': tf.placeholder(tf.float32),
                'learning_rate': tf.placeholder(tf.float32),
                'learning_rate_q': tf.placeholder(tf.float32),
                'ema_decay': tf.placeholder(tf.float32),
                'importance_weight': tf.placeholder(tf.float32,shape=(None,)),
                'weight_decay': tf.placeholder(tf.float32,shape=()),
                'entropy_loss_weight': tf.placeholder(tf.float32,shape=()),
            }
            self.inputs = inputs

            # build training networks

            # training network: policy
            # for input into Q_policy below
            with tf.variable_scope('policy'):
                policy_train, policy_train_logits, policy_convnet_h_train = self.policy_fn(inputs['state'],training=True)
                if self.discrete and self.straight_through_estimation:
                    policy_train_input = ste_prob_to_action(policy_train)  
                else:
                    policy_train_input = policy_train

            
            # for evaluation in the environment
            with tf.variable_scope('policy',reuse=True):
                policy_eval, _, _ = self.policy_fn(inputs['state'],training=False)

            # training network: Q
            # for computing TD (time-delay) learning loss
            with tf.variable_scope('Q'):
                Q_action_train = self.q_fn(inputs['state'],inputs['action'],training=True)

            # training network: Reward
            with tf.variable_scope('R'):
                R_action_train = self.q_fn(inputs['state'],inputs['action'],training=True)

            # for computing policy gradient w.r.t. Q(state,policy)
            with tf.variable_scope('Q',reuse=True):
                Q_policy_train = self.q_fn(inputs['state'],policy_train_input,training=False)

            # target networks
            ema, ema_op, ema_vars_getter = exponential_moving_average(
                    scope.name,decay=inputs['ema_decay'],zero_debias=True)

            with tf.variable_scope('policy',reuse=True,custom_getter=ema_vars_getter):
                policy_ema_probs, _, _ = self.policy_fn(inputs['state'],training=False)
                if self.discrete:
                    pe_depth = self.action_shape[-1] 
                    pe_indices = tf.argmax(policy_ema_probs,axis=-1)
                    policy_ema = tf.one_hot(pe_indices,pe_depth)
                else:
                    policy_ema = policy_ema_probs

            with tf.variable_scope('policy',reuse=True,custom_getter=ema_vars_getter):
                policy_ema_state2_probs, _, _ = self.policy_fn(inputs['state2'],training=False)
                if self.discrete:
                    pe_depth = self.action_shape[-1] 
                    pe_indices = tf.argmax(policy_ema_state2_probs,axis=-1)
                    policy_ema_state2 = tf.one_hot(pe_indices,pe_depth)
                else:
                    policy_ema_state2 = policy_ema_state2_probs

            #with tf.variable_scope('Q',reuse=True,custom_getter=ema_vars_getter):
            q_custom_getter = None if self.stable else ema_vars_getter
            with tf.variable_scope('Q',reuse=True,custom_getter=q_custom_getter):
                Q_ema_state2 = self.q_fn(inputs['state2'],policy_ema_state2,training=False)

            with tf.variable_scope('R',reuse=True,custom_getter=ema_vars_getter):
                R_ema = self.q_fn(inputs['state'],policy_ema,training=False)

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

                if self.add_return_loss:
                    return_losses_Q = 0.5*tf.square(Q_action_train - inputs['returns'])
                    losses_Q += return_losses_Q

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
            losses_policy = dpg(Q_policy_train,policy_train_input,self.dqda_clipping,self.clip_norm)
            loss_policy = tf.reduce_mean(self.inputs['importance_weight']*losses_policy)
            loss = tf.reduce_mean(self.inputs['importance_weight']*losses_policy)

            # policy gradient
            policy_gradient = tf.gradients(losses_policy,policy_train)[0]
            print('policy_gradient: ',policy_gradient)

            with tf.variable_scope('Q') as scope_Q:
                self.var_list_Q = tf.trainable_variables(scope=scope_Q.name)
                if self.stable:
                    # stable gradients for parameters of Q
                    if self.opt_q_layerwise:
                        grad_Q = []
                        for v in self.var_list_Q:
                            if 'kernel' in v.name:
                                gv, _ = get_modified_gradients_pinv(
                                    [v],
                                    Q_action_train,
                                    Q_ema_state2,
                                    td_error,
                                    alpha=self.alpha,
                                    beta=self.beta,
                                    weight_decay=inputs['weight_decay'],
                                    online=self.opt_stable_q_online,
                                    online_decay=self.opt_stable_q_online_momentum,
                                )
                                grad_Q.extend(gv)

                    else:
                        grad_Q, _ = get_modified_gradients_pinv(
                            self.var_list_Q,
                            Q_action_train,
                            Q_ema_state2,
                            td_error,
                            alpha=self.alpha,
                            beta=self.beta,
                            weight_decay=inputs['weight_decay'],
                            online=self.opt_stable_q_online,
                            online_decay=self.opt_stable_q_online_momentum,
                        )
                else:
                    loss_Q = tf.reduce_mean(losses_Q) + l2_loss(self.var_list_Q,inputs['weight_decay'])
                    grad_Q = zip(tf.gradients(loss_Q,self.var_list_Q),self.var_list_Q)


            qkw = {} if self.optimizer_q_kwargs is None else self.optimizer_q_kwargs
            if self.optimizer_q == 'gradient_descent':
                self.optimizer_Q = tf.train.GradientDescentOptimizer(inputs['learning_rate_q'],**qkw)
            elif self.optimizer_q == 'momentum':
                self.optimizer_Q = tf.train.MomentumOptimizer(inputs['learning_rate_q'],**qkw)
            elif self.optimizer_q == 'rms_prop':
                self.optimizer_Q = tf.train.RMSPropOptimizer(inputs['learning_rate_q'],**qkw)
            elif self.optimizer_q == 'adam':
                self.optimizer_Q = tf.train.AdamOptimizer(inputs['learning_rate_q'],**qkw)
            else:
                raise NotImplementedError('optimizer_q="%s" not implemented' % self.optimizer_q)
            train_op_Q = self.optimizer_Q.apply_gradients(grad_Q)
            
            # gradient update for parameters of policy
            with tf.variable_scope('policy') as scope_policy:
                self.var_list_policy = tf.trainable_variables(scope=scope_policy.name)
            self.optimizer = tf.train.RMSPropOptimizer(inputs['learning_rate']) 

            # weight decay for policy loss
            if self.regularize_policy:
                var_list_policy_connected = get_connected_vars(self.var_list_policy,loss)
                l2_reg = inputs['weight_decay']*tf.reduce_sum([tf.nn.l2_loss(v) for v in var_list_policy_connected])
                loss += l2_reg

            loss += inputs['entropy_loss_weight']*tf.reduce_mean(entropy_loss(policy_train_logits))
            
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

            pnorms_policy = {v.name: tf.sqrt(tf.reduce_mean(tf.square(v))) for v in self.var_list_policy}
            pnorms_Q = {v.name: tf.sqrt(tf.reduce_mean(tf.square(v))) for v in self.var_list_Q}
            pnorm_policy = tf.linalg.global_norm(self.var_list_policy)
            pnorm_Q = tf.linalg.global_norm(self.var_list_Q)

            gradients_policy = tf.gradients(loss,self.var_list_policy)
            gnorm_policy = tf.linalg.global_norm(gradients_policy)

            # store attributes for later use
            self.outputs = {
                'y': y,
                'td_error': td_error,
                'loss': loss,
                'losses_Q': losses_Q,
                'losses_policy': losses_policy,
                'policy_train': policy_train,
                'policy_train_input': policy_train_input,
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
                'pnorms_policy': pnorms_policy,
                'pnorms_Q': pnorms_Q,
                'pnorm_policy': pnorm_policy,
                'pnorm_Q': pnorm_Q,
                'gnorm_policy': gnorm_policy,
                'gradients_policy': gradients_policy,
                'policy_convnet_h_train': policy_convnet_h_train,
            }

            if not self.episodic:
                self.outputs['reward_differential'] = reward_differential

    def get_feed_dict(self,inputs,placeholders=None):
        feed_dict = {}
        def func(inputs,placeholders):
            if isinstance(inputs,dict):
                for k in inputs:
                    func(inputs[k],placeholders[k])
            else:
                feed_dict[placeholders] = inputs
        placeholders = self.inputs if placeholders is None else placeholders
        func(inputs,placeholders)
        return feed_dict
        
    def act(self,state,session=None):
        session = session or tf.get_default_session()
        feed_dict = self.get_feed_dict(state,self.inputs['state'])
        return session.run(self.outputs['policy_eval'],feed_dict)
        
    def act_train(self,state,session=None):
        session = session or tf.get_default_session()
        feed_dict = self.get_feed_dict(state,self.inputs['state'])
        return session.run(self.outputs['policy_train'],feed_dict)

    def get_inputs(self,**inputs):
        return {self.inputs[k]: inputs[k] for k in inputs}

    def update(self,state,action,reward,done,state2,gamma=0.99,learning_rate=1e-3,learning_rate_q=1.,ema_decay=0.999,weight_decay=0.1,entropy_loss_weight=0.0,importance_weight=None,session=None,outputs=['td_error'],returns=None):
        session = session or tf.get_default_session()
        if importance_weight is None:
            importance_weight = np.ones_like(reward)
        inputs = {
            self.inputs['action']:action,
            self.inputs['reward']:reward,
            self.inputs['done']:done,
            self.inputs['gamma']:gamma,
            self.inputs['learning_rate']:learning_rate,
            self.inputs['learning_rate_q']:learning_rate_q,
            self.inputs['ema_decay']:ema_decay,
            self.inputs['weight_decay']:weight_decay,
            self.inputs['importance_weight']:importance_weight,
            self.inputs['entropy_loss_weight']:entropy_loss_weight,
        }
        state_inputs = self.get_feed_dict(state,self.inputs['state'])
        state2_inputs = self.get_feed_dict(state2,self.inputs['state2'])
        inputs.update(state_inputs)
        inputs.update(state2_inputs)

        if self.add_return_loss:
            inputs[self.inputs['returns']] = returns
        my_outputs, _ = session.run(
            [[self.outputs[k] for k in outputs],self.update_ops],
            inputs
        )
        return my_outputs
