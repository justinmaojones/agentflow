import tensorflow as tf
import numpy as np
from ..objectives import dpg, td_learning
from ..tensorflow.ops import exponential_moving_average, get_gradient_matrix, get_connected_vars, entropy_loss
from tensorflow.python.ops import gen_linalg_ops

def score_to_onehot(probs,axis=-1):
    depth = probs.shape[axis]
    argmax = tf.argmax(probs,axis)
    return tf.one_hot(argmax,depth)

def noisy_policy_categorical(policy,eps):
    shape = tf.shape(policy)
    batchsize = shape[0]
    depth = shape[-1]
    dims = len(policy.shape)
    random_policy = score_to_onehot(tf.random.uniform(shape),axis=-1)
    noise = tf.reshape(tf.random.uniform((batchsize,1)),[batchsize]+[1]*(dims-1))
    w = tf.cast(noise>eps,tf.float32)
    return w*policy + (1-w)*random_policy

def noisy_policy_gaussian(policy,std_dev):
    return policy + tf.random.normal(tf.shape(policy),0.0,std_dev)

def ste_prob_to_action(probs,axis=-1):
    """
    straight through gradient estimator
    """
    onehot = score_to_onehot(probs,axis)
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

def simple_modified_grad(g1,g2,err,alpha):
    g1g1 = tf.reduce_sum(tf.square(g1),axis=-1,keepdims=True)
    g1g2 = tf.reduce_sum(g1*g2,axis=-1,keepdims=True)
    g2g2 = tf.reduce_sum(tf.square(g2),axis=-1,keepdims=True)
    a = g1g1 + alpha
    b = g1g2
    c = g1g2
    d = g2g2 + alpha
    det = a*d-b*c
    return (g1*d - g2*g1g2)*(err[...,None]/(det+1e-12))

def unit_vector(x,axis=-1,keepdims=True):
    x_norm = tf.sqrt(tf.reduce_sum(tf.square(x),axis=axis,keepdims=keepdims))
    return x / (x_norm + 1e-8)

def vector_projection(x,y,axis=-1):
    """
    project x onto y
    """
    y_unit = unit_vector(y,axis,keepdims=True)
    return tf.reduce_sum(x*y_unit,axis=axis,keepdims=True)*y_unit

def vector_rejection(x,y,axis=-1):
    return x - vector_projection(x,y,axis=axis)

def orthogonal_gradient_descent(var_list,losses,y2_pred,weight_decay=None,ogd_weight=1.):
    var_list, gradients = get_gradient_matrix(var_list,losses)
    var_list2, gradients2 = get_gradient_matrix(var_list,y2_pred)

    modified_grad_flat = vector_rejection(gradients,gradients2)
    modified_grad_flat = tf.reduce_mean(modified_grad_flat,axis=0)

    g0 = tf.reduce_sum(gradients,axis=0)
    modified_grad_flat = ogd_weight*modified_grad_flat + (1-ogd_weight)*g0

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

def get_modified_gradients_pinv(var_list,y_pred,y2_pred,td_err,alpha,beta,q_decay=0,vprev=None,fast=True,weight_decay=None,normalize_gradients=False,grad_norm_clipping=None,online=False,online_decay=0.99,online2=False,simplified=False,simplified2=False):
    var_list, gradients = get_gradient_matrix(var_list,y_pred)
    var_list2, gradients2 = get_gradient_matrix(var_list,y2_pred)

    for v,v2 in zip(var_list,var_list2):
        assert v == v2

    A = gradients
    b = td_err
    if simplified:
        assert beta > 0
        assert q_decay > 0
        A = tf.concat([A,gradients2*(beta**0.5)],axis=1)
        A = tf.concat([A,gradients*(q_decay**0.5)],axis=1)
        zeros = tf.zeros_like(b)
        b = tf.concat([
            tf.reshape(b,(1,-1),name='reshape_b'),
            tf.reshape(zeros,(1,-1),name='reshape_zeros_beta'),
            tf.reshape(zeros,(1,-1),name='reshape_zeros_q_decay'),
        ],axis=1)
        A = tf.reshape(A,(-1,2,tf.shape(gradients)[-1]),name='reshape_A')
        b = tf.reshape(b,(-1,2),name='reshape_b2')
    else:
        if beta > 0:
            A = tf.concat([A,gradients2*(beta**0.5)],axis=0)
            zeros = tf.zeros(tf.shape(td_err)[0],tf.float32)
            b = tf.concat([b,zeros],axis=0)
        if q_decay > 0:
            A = tf.concat([A,gradients*(q_decay**0.5)],axis=0)
            b = tf.concat([b,y_pred],axis=0)
        if vprev is not None:
            A = tf.concat([A,vprev[None]],axis=0)
            zeros = tf.zeros(1,tf.float32)
            b = tf.concat([b,zeros],axis=0)

    if normalize_gradients:
        A_norm = tf.sqrt(tf.reduce_max(tf.reduce_sum(tf.square(A),axis=-1)))
        A = A/A_norm
        b = b/A_norm

    if grad_norm_clipping is not None:
        A_norm = tf.sqrt(tf.reduce_max(tf.reduce_sum(tf.square(A),axis=-1)))
        multiplier = grad_norm_clipping/A_norm
        A = tf.where(A_norm > grad_norm_clipping, A*multiplier, A)
        b = tf.where(A_norm > grad_norm_clipping, b*multiplier, b)

    if online:
        modified_grad_flat = online_lstsq(A,b[...,None],alpha,online_decay)
    elif online2:
        modified_grad_flat = online_lstsq2(A,b[...,None],alpha)
    elif simplified2:
        modified_grad_flat = simple_modified_grad(gradients,gradients2,td_err,alpha)
        modified_grad_flat = tf.reshape(tf.reduce_mean(modified_grad_flat,axis=0),[-1])
    else:
        modified_grad_flat = tf.linalg.lstsq(A,b[...,None],fast=fast,l2_regularizer=alpha)
        if simplified:
            modified_grad_flat = tf.reshape(tf.reduce_mean(modified_grad_flat,axis=0),[-1])
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
            clip_norm=False,discrete=False,episodic=True,beta=1,alpha=1,q_decay=0,q2_noise=False,
            optimizer_q='gradient_descent',opt_q_layerwise=False,optimizer_q_kwargs=None,
            regularize_policy=True,straight_through_estimation=False,
            add_return_loss=False,stable=True,stable_ema=False,grad_norm_clipping=None,
            opt_stable_q_online=False,opt_stable_q_online_momentum=0.99,
            q_normalized=False,noisy_target=0.0,td_loss_type='square',
            ogd=False,
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
        self.q_decay = q_decay
        self.q2_noise = q2_noise
        self.optimizer_q = optimizer_q
        self.opt_q_layerwise = opt_q_layerwise
        self.optimizer_q_kwargs = optimizer_q_kwargs
        self.regularize_policy = regularize_policy
        self.straight_through_estimation = straight_through_estimation
        self.add_return_loss = add_return_loss
        self.stable = stable
        self.stable_ema = stable_ema
        self.grad_norm_clipping = grad_norm_clipping
        self.opt_stable_q_online = opt_stable_q_online
        self.opt_stable_q_online_momentum = opt_stable_q_online_momentum
        self.q_normalized = q_normalized
        self.noisy_target = noisy_target
        self.td_loss_type = td_loss_type
        self.ogd = ogd

        self.build_model()

    def build_placeholder(self,tf_type,shape,name):
        if isinstance(shape,dict):
            return {k: self.build_placeholder(tf_type,shape[k],name=name+'/'+k)}
        else:
            return tf.placeholder(tf_type,shape=tuple([None]+shape),name=name)


    def build_model(self):

        with tf.variable_scope(None,default_name='DDPG') as scope:

            # inputs
            inputs = {
                'state': self.build_placeholder(tf.float32, self.state_shape, name='state'),
                'action': tf.placeholder(tf.float32,shape=tuple([None]+self.action_shape), name='action'),
                'reward': tf.placeholder(tf.float32,shape=(None,), name='reward'),
                'returns': tf.placeholder(tf.float32,shape=(None,), name='returns'),
                'done': tf.placeholder(tf.float32,shape=(None,), name='done'),
                'state2': self.build_placeholder(tf.float32, self.state_shape, name='state2'),
                'gamma': tf.placeholder(tf.float32,name='gamma'),
                'learning_rate': tf.placeholder(tf.float32,name='learning_rate'),
                'learning_rate_q': tf.placeholder(tf.float32,name='learning_rate_q'),
                'ema_decay': tf.placeholder(tf.float32,name='ema_decay',),
                'importance_weight': tf.placeholder(tf.float32,shape=(None,),name='importance_weight'),
                'weight_decay': tf.placeholder(tf.float32,shape=(),name='weight_decay'),
                'entropy_loss_weight': tf.placeholder(tf.float32,shape=(),name='entropy_loss_weight'),
                'alpha': tf.placeholder(tf.float32,shape=(),name='alpha'),
                'ogd_weight': tf.placeholder(tf.float32,shape=(),name='ogd_weight'),
            }
            self.inputs = inputs

            # build training networks

            # training network: policy
            # for input into Q_policy below
            with tf.variable_scope('policy'):
                policy_train, policy_train_logits, policy_convnet_h_train = self.policy_fn(inputs['state'],training=True)
                if self.q_normalized:
                    policy_train_input = policy_train_logits
                elif self.discrete and self.straight_through_estimation:
                    policy_train_input = ste_prob_to_action(policy_train)  
                else:
                    policy_train_input = policy_train

            
            # for evaluation in the environment
            with tf.variable_scope('policy',reuse=True):
                policy_eval, _, _ = self.policy_fn(inputs['state'],training=False)
                if self.discrete:
                    policy_eval_input = score_to_onehot(policy_eval)  
                else:
                    policy_eval_input = policy_eval

            # training network: Q
            # for computing TD (time-delay) learning loss
            with tf.variable_scope('Q'):
                Q_action_train = self.q_fn(inputs['state'],inputs['action'],training=True)

            with tf.variable_scope('Q',reuse=True):
                Q_action_eval = self.q_fn(inputs['state'],inputs['action'],training=False,evaluate=True)

            # training network: Reward
            with tf.variable_scope('R'):
                R_action_train = self.q_fn(inputs['state'],inputs['action'],training=True)

            # for computing policy gradient w.r.t. Q(state,policy)
            with tf.variable_scope('Q',reuse=True):
                Q_policy_train = self.q_fn(inputs['state'],policy_train_input,training=True,q_normalized=self.q_normalized)

            with tf.variable_scope('Q',reuse=True):
                Q_policy_eval = self.q_fn(inputs['state'],policy_eval_input,training=False,evaluate=True)

            # target networks
            ema, ema_op, ema_vars_getter = exponential_moving_average(
                    scope.name,decay=inputs['ema_decay'],zero_debias=True)

            with tf.variable_scope('policy',reuse=True,custom_getter=ema_vars_getter):
                policy_ema_probs, _, _ = self.policy_fn(inputs['state'],training=False)
                if self.discrete:
                    policy_ema = score_to_onehot(policy_ema_probs)
                else:
                    policy_ema = policy_ema_probs

            with tf.variable_scope('policy',reuse=True,custom_getter=ema_vars_getter):
                policy_ema_state2_probs, _, _ = self.policy_fn(inputs['state2'],training=False)
                if self.discrete:
                    policy_ema_state2 = score_to_onehot(policy_ema_state2_probs)
                    if self.noisy_target > 0:
                        policy_ema_state2 = noisy_policy_categorical(policy_ema_state2,self.noisy_target)
                else:
                    policy_ema_state2 = policy_ema_state2_probs
                    if self.noisy_target > 0:
                        policy_ema_state2 = noisy_policy_gaussian(policy_ema_state2,self.noisy_target)

            #with tf.variable_scope('Q',reuse=True,custom_getter=ema_vars_getter):
            with tf.variable_scope('Q',reuse=True):
                Q_state2 = self.q_fn(inputs['state2'],policy_ema_state2,training=False)

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
            Q_state2 = tf.reshape(Q_state2,[-1])
            Q_ema_state2 = tf.reshape(Q_ema_state2,[-1])
            R_action_train = tf.reshape(R_action_train,[-1])
            R_ema = tf.reshape(R_ema,[-1])

            # average reward
            reward_avg = tf.Variable(tf.zeros(1),dtype=tf.float32,name='avg_reward')

            # loss functions
            if self.episodic:
                if self.stable:
                    if self.stable_ema:
                        Q2 = Q_ema_state2
                    else:
                        Q2 = Q_state2
                else:
                    Q2 = Q_ema_state2
                if self.q2_noise:
                    Q2 += tf.random.normal(tf.shape(Q2),stddev=self.q_decay)

                losses_Q, y, td_error = td_learning(
                        Q_action_train,
                        reward,
                        inputs['gamma'],
                        (1-done)*Q2,
                        loss=self.td_loss_type,
                    )

                if self.add_return_loss:
                    return_losses_Q = 0.5*tf.square(Q_action_train - inputs['returns'])
                    losses_Q += return_losses_Q

                loss_R = 1.
            else:
                raise NotImplementedError("THIS DOESN'T WORK")
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
            policy_gradient_raw = tf.gradients(Q_policy_train,policy_train_input)[0]
            policy_gradient_raw2 = tf.gradients(dpg(Q_policy_train,policy_train_input),policy_train)[0]
            policy_gradient = tf.gradients(losses_policy,policy_train_input)[0]
            policy_gradient_logits = tf.gradients(losses_policy,policy_train_logits)[0]
            policy_gradient_conv_h = tf.gradients(losses_policy,policy_convnet_h_train)[0]
            print('policy_gradient: ',policy_gradient)
            print('policy_gradient_logits: ',policy_gradient_logits)

            with tf.variable_scope('Q') as scope_Q:
                self.var_list_Q = tf.trainable_variables(scope=scope_Q.name)
                self.moving_avg_var_list_Q = tf.moving_average_variables(scope=scope_Q.name)
                if self.stable:
                    # stable gradients for parameters of Q
                    if self.opt_q_layerwise:
                        grad_Q = []
                        for v in self.var_list_Q:
                            if 'kernel' in v.name:
                                gv, _ = get_modified_gradients_pinv(
                                    [v],
                                    Q_action_train,
                                    Q_state2,
                                    td_error,
                                    alpha=inputs['alpha'],
                                    beta=self.beta,
                                    q_decay=self.q_decay,
                                    weight_decay=inputs['weight_decay'],
                                    grad_norm_clipping=self.grad_norm_clipping,
                                    online=self.opt_stable_q_online,
                                    online_decay=self.opt_stable_q_online_momentum,
                                )
                                grad_Q.extend(gv)

                    else:
                        grad_Q, _ = get_modified_gradients_pinv(
                            self.var_list_Q,
                            Q_action_train,
                            Q_state2,
                            td_error,
                            alpha=inputs['alpha'],
                            beta=self.beta,
                            q_decay=self.q_decay,
                            weight_decay=inputs['weight_decay'],
                            grad_norm_clipping=self.grad_norm_clipping,
                            online=self.opt_stable_q_online,
                            online_decay=self.opt_stable_q_online_momentum,
                        )
                elif self.ogd:
                    losses_Q = tf.reshape(losses_Q,tf.shape(Q_action_train))
                    grad_Q, _ = orthogonal_gradient_descent(
                        self.var_list_Q,
                        losses_Q,
                        (1-done)*Q_state2,
                        weight_decay = inputs['weight_decay'],
                        ogd_weight = inputs['ogd_weight'],
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
                self.moving_avg_var_list_policy = tf.moving_average_variables(scope=scope_policy.name)
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

            pnorms_mv_avg_policy = {v.name: tf.sqrt(tf.reduce_mean(tf.square(v))) for v in self.var_list_policy}
            pnorms_mv_avg_Q = {v.name: tf.sqrt(tf.reduce_mean(tf.square(v))) for v in self.var_list_Q}
            pnorm_mv_avg_policy = tf.linalg.global_norm(self.moving_avg_var_list_policy)
            pnorm_mv_avg_Q = tf.linalg.global_norm(self.moving_avg_var_list_Q)

            gradients_policy = tf.gradients(loss,self.var_list_policy)
            gnorm_policy = tf.linalg.global_norm(gradients_policy)

            policy_gradient_norm = tf.norm(policy_gradient,ord=2,axis=1)
            policy_gradient_logits_norm = tf.norm(policy_gradient_logits,ord=2,axis=1)
            policy_gradient_conv_h_norm = tf.norm(policy_gradient_conv_h,ord=2,axis=1)

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
                'Q_action_eval': Q_action_eval,
                'Q_policy_train': Q_policy_train,
                'Q_policy_eval': Q_policy_eval,
                'policy_ema_probs': policy_ema_probs,
                'policy_ema': policy_ema,
                'policy_ema_state2_probs': policy_ema_state2_probs,
                'policy_ema_state2': policy_ema_state2,
                'Q_ema_state2': Q_ema_state2,
                'Q2': Q2,
                'R_action_train': R_action_train,
                'R_ema': R_ema,
                'reward_avg': reward_avg,
                'policy_gradient_raw': policy_gradient_raw,
                'policy_gradient_raw2': policy_gradient_raw2,
                'policy_gradient': policy_gradient,
                'policy_gradient_norm': policy_gradient_norm,
                'policy_gradient_logits': policy_gradient_logits,
                'policy_gradient_logits_norm': policy_gradient_logits_norm,
                'policy_gradient_conv_h': policy_gradient_conv_h,
                'policy_gradient_conv_h_norm': policy_gradient_conv_h_norm,
                'pnorms_policy': pnorms_policy,
                'pnorms_Q': pnorms_Q,
                'pnorm_policy': pnorm_policy,
                'pnorm_Q': pnorm_Q,
                'pnorms_mv_avg_policy': pnorms_mv_avg_policy,
                'pnorms_mv_avg_Q': pnorms_mv_avg_Q,
                'pnorm_mv_avg_policy': pnorm_mv_avg_policy,
                'pnorm_mv_avg_Q': pnorm_mv_avg_Q,
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
        
    def act(self,state,session=None,addl_outputs=[]):
        session = session or tf.get_default_session()
        feed_dict = self.get_feed_dict(state,self.inputs['state'])
        outputs = self.outputs['policy_eval']
        if len(addl_outputs) > 0:
            outputs = [outputs] + [self.outputs[k] for k in addl_outputs]
        return session.run(outputs,feed_dict)
        
    def act_train(self,state,session=None):
        session = session or tf.get_default_session()
        feed_dict = self.get_feed_dict(state,self.inputs['state'])
        return session.run(self.outputs['policy_train'],feed_dict)

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

    def update(self,state,action,reward,done,state2,gamma=0.99,learning_rate=1e-3,learning_rate_q=1.,ema_decay=0.999,weight_decay=0.1,entropy_loss_weight=0.0,importance_weight=None,session=None,outputs=['td_error'],returns=None,alpha=None,ogd_weight=1.):
        session = session or tf.get_default_session()
        if importance_weight is None:
            importance_weight = np.ones_like(reward)
        if alpha is None:
            alpha = self.alpha
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
            self.inputs['alpha']:alpha,
            self.inputs['ogd_weight']:ogd_weight,
        }
        state_inputs = self.get_feed_dict(state,self.inputs['state'])
        state2_inputs = self.get_feed_dict(state2,self.inputs['state2'])
        inputs.update(state_inputs)
        inputs.update(state2_inputs)

        if self.add_return_loss:
            inputs[self.inputs['returns']] = returns
        my_outputs, _ = session.run(
            [{k:self.outputs[k] for k in outputs},self.update_ops],
            inputs
        )
        return my_outputs
