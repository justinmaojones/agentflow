from agentflow.env import VecGymEnv
from agentflow.agents import StableDDPG
from agentflow.buffers import BufferMap, PrioritizedBufferMap
from agentflow.state import NPrevFramesStateEnv
from agentflow.state import AddEpisodeTimeStateEnv
from agentflow.state import ResizeImageStateEnv
from agentflow.state import CvtRGB2GrayImageStateEnv
from agentflow.numpy.ops import onehot
from agentflow.tensorflow.nn import dense_net
from agentflow.tensorflow.ops import normalize_ema
from agentflow.utils import check_whats_connected
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

def conv_net(x, conv_units, fc_units, fc_layers, batchnorm=True,activation=tf.nn.relu,training=False,**kwargs):

    conv_layers = 3
    assert isinstance(conv_layers,int) and conv_layers > 0, 'conv_layers should be a positive integer'
    assert isinstance(conv_units, int) and conv_units > 0, 'conv_units should be a positive integer'
    assert isinstance(fc_layers,int) and fc_layers > 0, 'fc_layers should be a positive integer'
    assert isinstance(fc_units, int) and fc_units > 0, 'fc_units should be a positive integer'

    h = x
    for l in range(conv_layers):
        kernel_size = (3,3)
        strides = (2,2)
        h = tf.layers.conv2d(h,conv_units,kernel_size,strides)
        h = activation(h)
        #h = tf.layers.max_pooling2d(h,(2,2),(2,2))

        if batchnorm:
            BN = tf.layers.BatchNormalization()
            h = BN(h,training=training)

    h = tf.layers.flatten(h)
    h = dense_net(h,fc_units,fc_layers,batchnorm,activation,training,**kwargs)
    return h

def build_net_fn(hidden_dims,hidden_layers,output_dim,batchnorm):
    def net_fn(h,training=False):
        h = dense_net(h,hidden_dims,hidden_layers,batchnorm=batchnorm,training=training)
        return tf.layers.dense(h,output_dim)
    return net_fn

def build_conv_net_fn(hidden_dims,hidden_layers,output_dim,batchnorm,stop_gradient=False):
    def conv_net_fn(h,training=False):
        h = conv_net(h,hidden_dims,hidden_dims,hidden_layers,batchnorm=batchnorm,training=training)
        h = tf.layers.dense(h,output_dim)
        if stop_gradient:
            h = tf.stop_gradient(h)
        return h
    return conv_net_fn

def build_policy_fn(hidden_dims,hidden_layers,output_dim,batchnorm,normalize_inputs=True,freeze_conv_net=False):
    dense_net_fn = build_net_fn(hidden_dims,hidden_layers,output_dim,batchnorm)
    conv_net_fn = build_conv_net_fn(hidden_dims,hidden_layers,output_dim,batchnorm,freeze_conv_net)
    def policy_fn(state,training=False):
        state = state/255. - 0.5
        if normalize_inputs:
            state, _ = normalize_ema(state,training)
        h = conv_net_fn(state,training,)
        h = dense_net_fn(h,training)
        return tf.nn.softmax(h,axis=-1)
    return policy_fn

def build_q_fn(hidden_dims,hidden_layers,output_dim,batchnorm,normalize_inputs=True,freeze_conv_net=False):
    dense_net_fn = build_net_fn(hidden_dims,hidden_layers,output_dim,batchnorm)
    conv_net_fn = build_conv_net_fn(hidden_dims,hidden_layers,hidden_dims,batchnorm,freeze_conv_net)
    def q_fn(state,action,training=False):
        state = state/255. - 0.5
        if normalize_inputs:
            state, _ = normalize_ema(state,training)
        h_state = conv_net_fn(state,training)
        h_action = tf.layers.dense(action,hidden_dims)
        h = tf.concat([h_state,h_action],axis=1)
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

def entropy(a):
    p = (a[:,None] == np.unique(a)[None]).mean(axis=0)
    return -(p*np.log(p)).sum()

def noisy_action(action_softmax,eps=1.,clip=5e-2):
    action_softmax_clipped = np.clip(action_softmax,clip,1-clip)
    logit_unscaled = np.log(action_softmax_clipped)
    u = np.random.rand(*logit_unscaled.shape)
    g = -np.log(-np.log(u))
    return (eps*g+logit_unscaled).argmax(axis=-1)

class TrackEpisodeScore(object):

    def __init__(self):
        self._prev_ep_scores = None
        self._curr_ep_scores = None

    def get_prev_ep_scores(self):
        assert self._prev_ep_scores is not None
        return self._prev_ep_scores

    def update(self,rewards,dones):
        if self._curr_ep_scores is None:
            self._prev_ep_scores = np.zeros_like(rewards) 
            self._curr_ep_scores = np.zeros_like(rewards)
        self._curr_ep_scores += rewards
        self._prev_ep_scores = self._prev_ep_scores*(1-dones) + self._curr_ep_scores*dones
        self._curr_ep_scores *= 1-dones
        return self.get_prev_ep_scores()


@click.command()
@click.option('--num_steps', default=20000, type=int)
@click.option('--n_envs', default=1)
@click.option('--env_id', default='PongDeterministic-v4')
@click.option('--n_prev_frames', default=12, type=int)
@click.option('--dqda_clipping', default=1.)
@click.option('--clip_norm', default=True, type=bool)
@click.option('--ema_decay', default=0.99, type=float)
@click.option('--hidden_dims', default=32)
@click.option('--hidden_layers', default=2)
@click.option('--freeze_conv_net', default=False, type=bool)
@click.option('--output_dim', default=6)
@click.option('--normalize_inputs', default=True, type=bool)
@click.option('--batchnorm', default=False, type=bool)
@click.option('--add_episode_time_state', default=False, type=bool)
@click.option('--buffer_type', default='normal', type=click.Choice(['normal','prioritized']))
@click.option('--buffer_size', default=2**11, type=int)
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
@click.option('--learning_rate_q', default=1.)
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

    cfg_hash = str(hash(str(sorted(cfg))))
    savedir = os.path.join(cfg['savedir'],'experiment' + cfg_hash)
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
            cfg['normalize_inputs'],
            cfg['freeze_conv_net'],
    )

    q_fn = build_q_fn(
            cfg['hidden_dims'],
            cfg['hidden_layers'],
            1,
            cfg['batchnorm'],
            cfg['normalize_inputs'],
            cfg['freeze_conv_net'],
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
    )

    # Replay Buffer
    if cfg['buffer_type'] == 'prioritized':
        replay_buffer = PrioritizedBufferMap(
                cfg['buffer_size'],
                alpha=cfg['prioritized_replay_alpha'],
                eps=cfg['prioritized_replay_eps'])
    else:
        replay_buffer = BufferMap(cfg['buffer_size'])

    log = {
        'reward_history': [],
        'done_history': [],
        'action_history': [],
        'action_probs_history': [],
        'test_ep_returns': [],
        'test_ep_actions_entropy': [],
        'test_ep_steps': [],
        'test_ep_rewards': {},
        'test_ep_dones': {},
        'test_ep_actions': {},
        'train_ep_returns': [],
        'step_duration_sec': [],
        'duration_cumulative': [],
        'beta': [],
        'max_importance_weight': [],
        'Q': [],
        'loss_Q': [],
        'Q_updates': [],
        'loss_Q_updates': [],
        'frames': [],
    }
    track_train_ep_returns = TrackEpisodeScore()

    state = env.reset()

    start_time = time.time()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        T = cfg['num_steps']
        T_beta = T if cfg['prioritized_replay_beta_iters'] is None else cfg['prioritized_replay_beta_iters']
        beta0 = cfg['prioritized_replay_beta0']
        pb = tf.keras.utils.Progbar(T,stateful_metrics=[
            'test_ep_returns',
            'test_ep_actions_entropy',
            'test_ep_length',
            'avg_action',
            'Q_action_train',
            'losses_Q'
        ])
        for t in range(T):
            start_step_time = time.time()

            action_probs = agent.act(state)

            #if len(replay_buffer) < cfg['begin_learning_at_step'] or np.random.rand() <= 0.02:
            #    action = np.random.choice(action.shape[1],size=len(action))
            #else:
            #    action = action.argmax(axis=1)
            if len(replay_buffer) >= cfg['begin_learning_at_step']:
                action = noisy_action(action_probs)
            else:
                action = np.random.choice(action_probs.shape[1],size=len(action_probs))

            state2, reward, done, info = env.step(action.astype('int').ravel())

            log['frames'].append(len(state2))
            log['action_probs_history'].append(action_probs)
            log['train_ep_returns'].append(track_train_ep_returns.update(reward,done))

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

            pb_input = []
            if t >= cfg['begin_learning_at_step'] and t % cfg['update_freq'] == 0:
                Q_action_train_list = []
                losses_Q_list = []

                if t >= cfg['begin_learning_at_step'] + cfg['n_steps_train_only_q']:
                    policy_learning_rate = cfg['learning_rate']
                else:
                    policy_learning_rate = 0.

                for i in range(cfg['n_update_steps']):

                    if cfg['buffer_type'] == 'prioritized':

                        beta = beta0 + (1.-beta0)*min(1.,float(t)/T_beta)
                        log['beta'].append(beta)

                        sample = replay_buffer.sample(cfg['batchsize'],beta=beta)
                        if cfg['prioritized_replay_weights_uniform']:
                            sample['importance_weight'] = np.ones_like(sample['importance_weight'])
                        log['max_importance_weight'].append(sample['importance_weight'].max())

                        td_error, Q_action_train, losses_Q = agent.update(
                                learning_rate=policy_learning_rate,
                                learning_rate_q=cfg['learning_rate_q'],
                                ema_decay=cfg['ema_decay'],
                                gamma=cfg['gamma'],
                                weight_decay=cfg['weight_decay'],
                                outputs=['td_error','Q_action_train','losses_Q'],
                                **sample)

                        if not cfg['prioritized_replay_simple']:
                            replay_buffer.update_priorities(td_error)
                    else:

                        sample = replay_buffer.sample(cfg['batchsize'])

                        Q_action_train, losses_Q = agent.update(
                                learning_rate=policy_learning_rate,
                                learning_rate_q=cfg['learning_rate_q'],
                                ema_decay=cfg['ema_decay'],
                                gamma=cfg['gamma'],
                                weight_decay=cfg['weight_decay'],
                                outputs=['Q_action_train','losses_Q'],
                                **sample)
                    Q_action_train_list.append(Q_action_train.mean())
                    losses_Q_list.append(losses_Q.mean())
                    log['Q_updates'].append(Q_action_train_list[-1])
                    log['loss_Q_updates'].append(losses_Q_list[-1])
                Q_action_train_mean = np.mean(Q_action_train_list)
                losses_Q_mean = np.mean(losses_Q_list)
                pb_input.append(('Q_action_train', Q_action_train_mean))
                pb_input.append(('loss_Q', losses_Q_mean))
                log['Q'].append(Q_action_train_mean)
                log['loss_Q'].append(losses_Q_mean)

            # Bookkeeping
            log['reward_history'].append(reward)
            log['done_history'].append(done)
            log['action_history'].append(action)
            avg_action = np.mean(log['action_history'][-20:])
            pb_input.append(('avg_action', avg_action))

            if t % cfg['n_steps_per_eval'] == 0 or t==0:
                test_ep_returns, test_ep_rewards, test_ep_dones, test_ep_actions = test_agent(test_env,agent)
                test_ep_actions_entropy = entropy(test_ep_actions.ravel())
                test_ep_length = len(test_ep_actions)
                log['test_ep_returns'].append(test_ep_returns)
                log['test_ep_length'].append(test_ep_length)
                log['test_ep_actions_entropy'].append(test_ep_actions_entropy)
                log['test_ep_rewards'][t] = test_ep_rewards
                log['test_ep_dones'][t] = test_ep_dones
                log['test_ep_actions'][t] = test_ep_actions
                log['test_ep_steps'].append(t)
                #log['test_duration_cumulative'].append(time.time()-start_time)
                avg_test_ep_returns = np.mean(log['test_ep_returns'][-1:])
                pb_input.append(('test_ep_returns', avg_test_ep_returns))
                pb_input.append(('test_ep_actions_entropy', test_ep_actions_entropy))

            pb.add(1,pb_input)
            end_time = time.time()
            log['step_duration_sec'].append(end_time-start_step_time)
            log['duration_cumulative'].append(end_time-start_time)

            if t % cfg['n_steps_per_eval'] == 0 and t > 0:
                with h5py.File(os.path.join(savedir,'log_intermediate.h5'), 'w') as f:
                    for k in log:
                        if isinstance(log[k],dict):
                            g = f.create_group(k)
                            for k2 in log[k]:
                                g[str(k2)] = log[k][k2]
                        else:
                            f[k] = log[k]


    if cfg['buffer_type'] == 'prioritized':
        log['priorities'] = replay_buffer.priorities()
        log['sample_importance_weights'] = replay_buffer.sample(1000,beta=1.)['importance_weight']

    with h5py.File(os.path.join(savedir,'log.h5'), 'w') as f:
        print('SAVING TO: {savedir}'.format(**locals()))
        print('writing to h5 file')
        for k in log:
            print('H5: %s'%k)
            if isinstance(log[k],dict):
                g = f.create_group(k)
                for k2 in log[k]:
                    g[str(k2)] = log[k][k2]
            else:
                f[k] = log[k]

if __name__=='__main__':
    run()
