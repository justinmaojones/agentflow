from agentflow.env import VecGymEnv
from agentflow.agents import StableDDPG
from agentflow.buffers import BufferMap, PrioritizedBufferMap, NStepReturnPublisher
from agentflow.state import NPrevFramesStateEnv
from agentflow.state import AddEpisodeTimeStateEnv
from agentflow.state import ResizeImageStateEnv
from agentflow.state import CvtRGB2GrayImageStateEnv
from agentflow.numpy.ops import onehot
from agentflow.numpy.models import TrackEpisodeScore
from agentflow.tensorflow.nn import dense_net
from agentflow.tensorflow.ops import normalize_ema, binarize
from agentflow.utils import check_whats_connected, LogsTFSummary
import tensorflow as tf
import numpy as np
import h5py
import os
import yaml 
import time
import click

# TODO
# * modify preprocessing to include actions
# * how to handle time state?  it converts to float64 :(

def get_activation_fn(activation_choice_str):
    if activation_choice_str == 'relu':
        return tf.nn.relu
    elif activation_choice_str == 'elu':
        return tf.nn.elu
    elif activation_choice_str == 'prelu':
        return tf.nn.leaky_relu
    else:
        raise ValueError("unhandled activation type: %s" % activation_choice_str)

def conv_net(x, conv_units, batchnorm=True, activation=tf.nn.relu, training=False,**kwargs):

    conv_layers = 4 
    assert isinstance(conv_layers,int) and conv_layers > 0, 'conv_layers should be a positive integer'
    assert isinstance(conv_units, int) and conv_units > 0, 'conv_units should be a positive integer'

    units = conv_units

    h = x
    for l in range(conv_layers):
        kernel_size = (3,3)
        strides = (1,1)
        h = tf.layers.conv2d(h,units,kernel_size,strides)
        h = activation(h)
        h = tf.layers.max_pooling2d(h,(2,2),(2,2))

        if batchnorm:
            BN = tf.layers.BatchNormalization()
            h = BN(h,training=training)

    h = tf.layers.flatten(h)
    return h

def build_net_fn(hidden_dims,hidden_layers,output_dim,batchnorm,activation_fn):
    def net_fn(h,training=False):
        if batchnorm:
            BN = tf.layers.BatchNormalization()
            h = BN(h,training=training)
        h = dense_net(h,hidden_dims,hidden_layers,
                batchnorm=batchnorm,training=training,activation=activation_fn)
        return tf.layers.dense(h,output_dim)
    return net_fn

def build_conv_net_fn(conv_dims,batchnorm,activation_fn,stop_gradient=False):
    def conv_net_fn(h,training=False):
        if stop_gradient:
            _batchnorm = False
        else:
            _batchnorm = batchnorm
        h = conv_net(h,conv_dims,_batchnorm,activation_fn,training)
        if stop_gradient:
            h = tf.stop_gradient(h)
        return h
    return conv_net_fn

def preprocess_image_state(state,binarized,normalize_inputs,training):
    if binarized:
        state = binarize(state,8)
    else:
        state = state/255. - 0.5
        if normalize_inputs:
            state, _ = normalize_ema(state,training)
    return state

def build_policy_fn(hidden_dims,hidden_layers,output_dim,batchnorm,activation,normalize_inputs=True,freeze_conv_net=False,binarized=False,conv_dims=None):
    conv_dims = conv_dims if conv_dims is not None else hidden_dims
    activation_fn = get_activation_fn(activation)
    dense_net_fn = build_net_fn(hidden_dims,hidden_layers,output_dim,batchnorm,activation_fn)
    conv_net_fn = build_conv_net_fn(conv_dims,batchnorm,activation_fn,freeze_conv_net)
    def policy_fn(state,training=False):
        state = preprocess_image_state(state,binarized,normalize_inputs,training)
        h_convnet = conv_net_fn(state,training)
        logits = dense_net_fn(h_convnet,training)
        return tf.nn.softmax(logits,axis=-1), logits, h_convnet 
    return policy_fn

def build_q_fn(hidden_dims,hidden_layers,output_dim,batchnorm,activation,normalize_inputs=True,freeze_conv_net=False,binarized=False,conv_dims=None):
    conv_dims = conv_dims if conv_dims is not None else hidden_dims
    activation_fn = get_activation_fn(activation)
    dense_net_fn = build_net_fn(hidden_dims,hidden_layers,output_dim,batchnorm,activation_fn)
    conv_net_fn = build_conv_net_fn(conv_dims,batchnorm,activation_fn,freeze_conv_net)
    def q_fn(state,action,training=False):
        state = preprocess_image_state(state,binarized,normalize_inputs,training)
        h_state = conv_net_fn(state,training)
        h = tf.concat([h_state,action],axis=1)
        h = dense_net_fn(h,training)
        return h
    return q_fn

def test_agent(test_env,agent):
    state = test_env.reset()
    rt = None
    all_done = 0
    rewards = []
    dones = []
    actions = []
    while np.mean(all_done) < 1:
        action = agent.act(state).argmax(axis=-1).ravel()
        state, reward, done, _ = test_env.step(action)
        if rt is None:
            rt = reward.copy()
            all_done = done.copy()
        else:
            rt += reward*(1-all_done)
            all_done = np.maximum(done,all_done)
        rewards.append(reward)
        dones.append(done)
        actions.append(action)
    return rt, np.array(rewards), np.array(dones), np.array(actions)

def entropy(p,axis):
    return -(p*np.log(p+1e-12)).sum(axis=axis)

def empirical_entropy(a):
    p = (a[:,None] == np.unique(a)[None]).mean(axis=0)
    return entropy(p,axis=None)

def l2norm(x):
    return np.sum(x**2)**0.5

def noisy_action(action_softmax,eps=1.,clip=5e-2):
    action_softmax_clipped = np.clip(action_softmax,clip,1-clip)
    logit_unscaled = np.log(action_softmax_clipped)
    u = np.random.rand(*logit_unscaled.shape)
    g = -np.log(-np.log(u))
    return (eps*g+logit_unscaled).argmax(axis=-1)

def noisy_action(action_softmax,p=0.05):
    argmax = action_softmax.argmax(axis=-1)
    output_size = argmax.shape
    random = np.random.choice(action_softmax.shape[-1],size=output_size)
    eps = np.random.rand(*output_size)
    z = eps <= p
    return z*random + (1-z)*argmax

@click.option('--num_steps', default=20000, type=int)
@click.option('--n_envs', default=1)
@click.option('--env_id', default='PongDeterministic-v4')
@click.option('--n_prev_frames', default=12, type=int)
@click.option('--dqda_clipping', default=1.)
@click.option('--clip_norm', default=True, type=bool)
@click.option('--ema_decay', default=0.99, type=float)
@click.option('--conv_dims', default=None, type=int)
@click.option('--hidden_dims', default=32)
@click.option('--hidden_layers', default=2)
@click.option('--activation', default='relu', type=click.Choice(['relu','elu','prelu']))
@click.option('--binarized', default=False, type=bool)
@click.option('--freeze_conv_net', default=False, type=bool)
@click.option('--output_dim', default=6)
@click.option('--normalize_inputs', default=True, type=bool)
@click.option('--noisy_action_prob', default=0.05, type=float)
@click.option('--batchnorm', default=False, type=bool)
@click.option('--add_episode_time_state', default=False, type=bool)
@click.option('--buffer_type', default='normal', type=click.Choice(['normal','prioritized','delayed','delayed_prioritized']))
@click.option('--buffer_size', default=2**11, type=int)
@click.option('--enable_n_step_return_publisher', default=False, type=bool)
@click.option('--n_step_return', default=1, type=int)
@click.option('--prioritized_replay_alpha', default=0.6, type=float)
@click.option('--prioritized_replay_beta0', default=0.4, type=float)
@click.option('--prioritized_replay_beta_iters', default=None, type=int)
@click.option('--prioritized_replay_eps', default=1e-6, type=float)
@click.option('--prioritized_replay_weights_uniform', default=False, type=bool)
@click.option('--prioritized_replay_compute_init', default=False, type=bool)
@click.option('--prioritized_replay_simple', default=False, type=bool)
@click.option('--prioritized_replay_simple_reward_adder', default=16, type=float)
@click.option('--prioritized_replay_simple_done_adder', default=16, type=float)
@click.option('--begin_learning_at_step', default=1e4)
@click.option('--learning_rate', default=1e-4)
@click.option('--learning_rate_decay', default=1.)
@click.option('--learning_rate_q', default=1.)
@click.option('--learning_rate_q_decay', default=1.)
@click.option('--n_steps_train_only_q', default=0)
@click.option('--optimizer_q', type=str, default='gradient_descent')
@click.option('--optimizer_q_decay', type=float, default=None)
@click.option('--optimizer_q_momentum', type=float, default=None)
@click.option('--optimizer_q_use_nesterov', type=bool, default=None)
@click.option('--opt_q_layerwise', type=bool, default=False)
@click.option('--alpha', default=1.)
@click.option('--beta', default=1.)
@click.option('--gamma', default=0.99)
@click.option('--weight_decay', default=0.0)
@click.option('--regularize_policy', default=False, type=bool)
@click.option('--straight_through_estimation', default=False, type=bool)
@click.option('--entropy_loss_weight', default=0.0)
@click.option('--n_update_steps', default=4, type=int)
@click.option('--update_freq', default=1, type=int)
@click.option('--n_steps_per_eval', default=100, type=int)
@click.option('--batchsize', default=100)
@click.option('--savedir', default='results')
@click.option('--seed',default=None, type=int)
def run(**cfg):

    for k in sorted(cfg):
        print('CONFIG: ',k,str(cfg[k]))

    discrete = True

    if cfg['seed'] is not None:
        np.random.seed(cfg['seed'])
        tf.set_random_seed(int(10*cfg['seed']))

    ts = str(int(time.time()*1000))
    savedir = os.path.join(cfg['savedir'],'experiment' + ts)
    print('SAVING TO: {savedir}'.format(**locals()))
    os.system('mkdir -p {savedir}'.format(**locals()))

    with open(os.path.join(savedir,'config.yaml'),'w') as f:
        yaml.dump(cfg, f)

    # Environment
    gym_env = VecGymEnv(cfg['env_id'],n_envs=cfg['n_envs'])
    env = gym_env
    env = CvtRGB2GrayImageStateEnv(env)
    env = ResizeImageStateEnv(env,resized_shape=(64,64))
    env = NPrevFramesStateEnv(env,n_prev_frames=cfg['n_prev_frames'],flatten=False)

    test_env = VecGymEnv(cfg['env_id'],n_envs=1)
    test_env = CvtRGB2GrayImageStateEnv(test_env)
    test_env = ResizeImageStateEnv(test_env,resized_shape=(64,64))
    test_env = NPrevFramesStateEnv(test_env,n_prev_frames=cfg['n_prev_frames'],flatten=False)

    if cfg['add_episode_time_state']:
        print('ADDING EPISODE TIME STATES')
        assert False, "This will convert the state to float64, need a better way to handle this before using it"
        env = AddEpisodeTimeStateEnv(env) 
        test_env = AddEpisodeTimeStateEnv(test_env) 

    state = env.reset()
    state_shape = state.shape
    print('STATE SHAPE: ',state_shape)
    action_shape = gym_env.action_space().n 

    # Agent
    policy_fn = build_policy_fn(
            cfg['hidden_dims'],
            cfg['hidden_layers'],
            cfg['output_dim'],
            cfg['batchnorm'],
            cfg['activation'],
            cfg['normalize_inputs'],
            cfg['freeze_conv_net'],
            cfg['binarized'],
            cfg['conv_dims'],
    )

    q_fn = build_q_fn(
            cfg['hidden_dims'],
            cfg['hidden_layers'],
            1,
            cfg['batchnorm'],
            cfg['activation'],
            cfg['normalize_inputs'],
            cfg['freeze_conv_net'],
            cfg['binarized'],
            cfg['conv_dims'],
    )

    optimizer_q_kwargs = {}
    if cfg['optimizer_q_decay'] is not None:
        optimizer_q_kwargs['decay'] = cfg['optimizer_q_decay']
    if cfg['optimizer_q_momentum'] is not None:
        optimizer_q_kwargs['momentum'] = cfg['optimizer_q_momentum']
    if cfg['optimizer_q_use_nesterov'] is not None:
        optimizer_q_kwargs['use_nesterov'] = cfg['optimizer_q_use_nesterov']
    agent = StableDDPG(
        state_shape[1:],[action_shape],policy_fn,q_fn,
        cfg['dqda_clipping'],
        cfg['clip_norm'],
        discrete=discrete,
        alpha=cfg['alpha'],
        beta=cfg['beta'],
        optimizer_q=cfg['optimizer_q'],
        opt_q_layerwise=cfg['opt_q_layerwise'],
        optimizer_q_kwargs=optimizer_q_kwargs,
        regularize_policy=cfg['regularize_policy'],
        straight_through_estimation=cfg['straight_through_estimation'],
    )

    for v in tf.trainable_variables():
        print(v)

    # Replay Buffer
    if cfg['buffer_type'] == 'prioritized':
        replay_buffer = PrioritizedBufferMap(
                cfg['buffer_size'],
                alpha=cfg['prioritized_replay_alpha'],
                eps=cfg['prioritized_replay_eps'])
    elif cfg['buffer_type'] == 'delayed_prioritized':
        replay_buffer = DelayedPrioritizedBufferMap(
                max_length=cfg['buffer_size'],
                alpha=cfg['prioritized_replay_alpha'],
                eps=cfg['prioritized_replay_eps'])
    elif cfg['buffer_type'] == 'delayed':
        replay_buffer = DelayedBufferMap(cfg['buffer_size'])
    else:
        replay_buffer = BufferMap(cfg['buffer_size'])

    if cfg['enable_n_step_return_publisher']:
        replay_buffer = NStepReturnPublisher(
            replay_buffer,
            n_steps=cfg['n_step_return'],
            gamma=cfg['gamma'],
        )

    log = LogsTFSummary(savedir)
    track_train_ep_lengths = TrackEpisodeScore(gamma=1.)
    track_train_ep_returns = TrackEpisodeScore(gamma=1.)
    track_train_ep_returns_discounted = TrackEpisodeScore(gamma=cfg['gamma'])

    state = env.reset()

    start_time = time.time()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())

        T = cfg['num_steps']
        T_beta = T if cfg['prioritized_replay_beta_iters'] is None else cfg['prioritized_replay_beta_iters']
        beta0 = cfg['prioritized_replay_beta0']
        pb = tf.keras.utils.Progbar(T,stateful_metrics=[
            'train_ep_lengths',
            'train_ep_returns',
            'test_ep_returns',
            'test_ep_actions_entropy',
            'test_ep_length',
            'avg_action',
            'Q_action_train',
            'losses_Q'
        ])
        for t in range(T):
            pb_input = []

            start_step_time = time.time()

            action_probs = agent.act(state)

            #if len(replay_buffer) < cfg['begin_learning_at_step'] or np.random.rand() <= 0.02:
            #    action = np.random.choice(action.shape[1],size=len(action))
            #else:
            #    action = action.argmax(axis=1)
            if len(replay_buffer) >= cfg['begin_learning_at_step']:
                action = noisy_action(action_probs,p=cfg['noisy_action_prob'])
            else:
                action = np.random.choice(action_probs.shape[1],size=len(action_probs))

            state2, reward, done, info = env.step(action.astype('int').ravel())

            # Bookkeeping
            log.append('reward_history',reward)
            log.append('done_history',done)
            log.append('action_history',action)
            avg_action = np.mean(log['action_history'][-20:])
            pb_input.append(('avg_action', avg_action))
            avg_train_ep_lengths = np.mean(log['train_ep_lengths'][-1:])
            pb_input.append(('train_ep_lengths', avg_train_ep_lengths))
            avg_train_ep_returns = np.mean(log['train_ep_returns'][-1:])
            pb_input.append(('train_ep_returns', avg_train_ep_returns))

            log.append('frames',len(state2))
            log.append('action_probs_history',action_probs)
            log.append('train_action_entropy',empirical_entropy(np.array(log['action_history'][-20:]).ravel()))
            log.append('action_probs_entropy',entropy(action_probs,axis=1))
            log.append('train_ep_lengths',track_train_ep_lengths.update(np.ones_like(reward),done))
            log.append('train_ep_returns',track_train_ep_returns.update(reward,done))
            log.append('train_ep_returns_discounted',track_train_ep_returns_discounted.update(reward,done))

            data = {
                'state':state,
                'action':onehot(action,depth=action_shape),
                'reward':reward,
                'done':done,
                'state2':state2
            }
            if cfg['buffer_type'] == 'prioritized' and cfg['prioritized_replay_compute_init']:
                priority = sess.run(agent.outputs['td_error'],agent.get_inputs(gamma=cfg['gamma'],**data))
                replay_buffer.append(data,priority=priority)
            elif cfg['buffer_type'] == 'prioritized' and cfg['prioritized_replay_simple']:
                priority = np.ones_like(data['reward'])
                priority += cfg['prioritized_replay_simple_reward_adder']*(data['reward']!=0)
                priority += cfg['prioritized_replay_simple_done_adder']*data['done']
                replay_buffer.append(data,priority=priority)
            else:
                replay_buffer.append(data)
            state = state2 

            if t >= cfg['begin_learning_at_step'] and t % cfg['update_freq'] == 0:
                Q_action_train_list = []
                losses_Q_list = []

                if t >= cfg['begin_learning_at_step'] + cfg['n_steps_train_only_q']:
                    t2 = t - (cfg['begin_learning_at_step'] + cfg['n_steps_train_only_q'])
                    policy_learning_rate = cfg['learning_rate']*(cfg['learning_rate_decay']**t2)
                else:
                    policy_learning_rate = 0.

                tq = t - cfg['begin_learning_at_step']
                learning_rate_q = cfg['learning_rate_q']*(cfg['learning_rate_q_decay']**tq)

                log.append('learning_rate_policy',policy_learning_rate)
                log.append('learning_rate_q',learning_rate_q)

                for i in range(cfg['n_update_steps']):

                    if cfg['buffer_type'] == 'prioritized':

                        beta = beta0 + (1.-beta0)*min(1.,float(t)/T_beta)
                        log.append('beta',beta)

                        sample = replay_buffer.sample(cfg['batchsize'],beta=beta)
                        if cfg['prioritized_replay_weights_uniform']:
                            sample['importance_weight'] = np.ones_like(sample['importance_weight'])
                        log.append('max_importance_weight',sample['importance_weight'].max())

                        td_error, Q_action_train, losses_Q, pnorms_policy, pnorms_Q, policy_gradient, gnorm_policy, policy_convnet_h_train = agent.update(
                                learning_rate=policy_learning_rate,
                                learning_rate_q=learning_rate_q,
                                ema_decay=cfg['ema_decay'],
                                gamma=cfg['gamma'],
                                weight_decay=cfg['weight_decay'],
                                entropy_loss_weight=cfg['entropy_loss_weight'],
                                outputs=[
                                    'td_error','Q_action_train','losses_Q',
                                    'pnorms_policy','pnorms_Q',
                                    'policy_gradient','gnorm_policy',
                                    'policy_convnet_h_train',
                                    ],
                                **sample)

                        if not cfg['prioritized_replay_simple']:
                            replay_buffer.update_priorities(td_error)
                    else:

                        sample = replay_buffer.sample(cfg['batchsize'])

                        Q_action_train, losses_Q, pnorms_policy, pnorms_Q, policy_gradient, gnorm_policy, policy_convnet_h_train = agent.update(
                                learning_rate=policy_learning_rate,
                                learning_rate_q=learning_rate_q,
                                ema_decay=cfg['ema_decay'],
                                gamma=cfg['gamma'],
                                weight_decay=cfg['weight_decay'],
                                entropy_loss_weight=cfg['entropy_loss_weight'],
                                outputs=[
                                    'td_error','Q_action_train','losses_Q',
                                    'pnorms_policy','pnorms_Q',
                                    'policy_gradient','gnorm_policy',
                                    'policy_convnet_h_train',
                                    ],
                                **sample)
                    Q_action_train_list.append(Q_action_train.mean())
                    losses_Q_list.append(losses_Q.mean())
                    log.append('Q_updates',Q_action_train_list[-1])
                    log.append('loss_Q_updates',losses_Q_list[-1])
                    log.append('policy_convnet_h_train',policy_convnet_h_train)
                    log.append('policy_gradient',policy_gradient)
                log.append('policy_gradient_norm',l2norm(policy_gradient))
                log.append('gnorm_policy',gnorm_policy)
                for k in pnorms_policy:
                    log.append('pnorms_policy: %s'%k,pnorms_policy[k])
                for k in pnorms_Q:
                    log.append('pnorms_Q: %s'%k,pnorms_Q[k])
                Q_action_train_mean = np.mean(Q_action_train_list)
                losses_Q_mean = np.mean(losses_Q_list)
                pb_input.append(('Q_action_train', Q_action_train_mean))
                pb_input.append(('loss_Q', losses_Q_mean))
                log.append('Q',Q_action_train_mean)
                log.append('loss_Q',losses_Q_mean)


            if (t % cfg['n_steps_per_eval'] == 0 and t >= cfg['begin_learning_at_step']) or t==0:
                test_ep_returns, test_ep_rewards, test_ep_dones, test_ep_actions = test_agent(test_env,agent)
                test_ep_actions_entropy = empirical_entropy(test_ep_actions.ravel())
                test_ep_length = len(test_ep_actions)
                log.append('test_ep_returns',test_ep_returns)
                log.append('test_ep_length',test_ep_length)
                log.append('test_ep_actions_entropy',test_ep_actions_entropy)
                log.append('test_ep_rewards',test_ep_rewards)
                log.append('test_ep_dones',test_ep_dones)
                log.append('test_ep_actions',test_ep_actions)
                log.append('test_ep_steps',t)
                #log.append('test_duration_cumulative',time.time()-start_time)
                avg_test_ep_returns = np.mean(log['test_ep_returns'][-1:])
                pb_input.append(('test_ep_returns', avg_test_ep_returns))
                pb_input.append(('test_ep_length', test_ep_length))
                pb_input.append(('test_ep_actions_entropy', test_ep_actions_entropy))

            end_time = time.time()
            log.append('step_duration_sec',end_time-start_step_time)
            log.append('duration_cumulative',end_time-start_time)

            pb.add(1,pb_input)
            log.flush(step=t)
            if t % cfg['n_steps_per_eval'] == 0 and t > 0:
                log.write(os.path.join(savedir,'log_intermediate.h5'),verbose=False)


    if cfg['buffer_type'] == 'prioritized':
        log.append('priorities',replay_buffer.priorities())
        log.append('sample_importance_weights',replay_buffer.sample(100,beta=1.)['importance_weight'])

    log.write(os.path.join(savedir,'log.h5'))

if __name__=='__main__':
    click.command()(run)()
