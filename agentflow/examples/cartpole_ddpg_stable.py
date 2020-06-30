from agentflow.env import VecGymEnv
from agentflow.agents import StableDDPG
from agentflow.buffers import BufferMap, DelayedBufferMap, PrioritizedBufferMap, DelayedPrioritizedBufferMap
from agentflow.state import NPrevFramesStateEnv, AddEpisodeTimeStateEnv
from agentflow.numpy.ops import onehot
from agentflow.tensorflow.nn import dense_net
from agentflow.tensorflow.ops import normalize_ema
from agentflow.utils import check_whats_connected, LogsTFSummary
import tensorflow as tf
import numpy as np
import h5py
import os
import yaml 
import time
import click

def build_net_fn(hidden_dims,hidden_layers,output_dim,batchnorm):
    def net_fn(h,training=False):
        h = dense_net(h,hidden_dims,hidden_layers,batchnorm=batchnorm,training=training)
        return tf.layers.dense(h,output_dim)
    return net_fn

def build_policy_fn(hidden_dims,hidden_layers,output_dim,batchnorm,normalize_inputs=True):
    net_fn = build_net_fn(hidden_dims,hidden_layers,output_dim,batchnorm)
    def policy_fn(state,training=False):
        if normalize_inputs:
            state, _ = normalize_ema(state,training)
        logits = net_fn(state,training)
        return tf.nn.softmax(logits,axis=-1), logits, state 
    return policy_fn

def build_q_fn(hidden_dims,hidden_layers,output_dim,batchnorm,normalize_inputs=True):
    net_fn = build_net_fn(hidden_dims,hidden_layers,output_dim,batchnorm)
    def q_fn(state,action,training=False):
        if normalize_inputs:
            state, _ = normalize_ema(state,training)
        h = tf.concat([state,action],axis=1)
        return net_fn(h,training)
    return q_fn

def test_agent(test_env,agent):
    state = test_env.reset()
    rt = None
    all_done = 0
    while np.mean(all_done) < 1:
        action = agent.act(state).argmax(axis=-1).ravel()
        state, reward, done, _ = test_env.step(action)
        if rt is None:
            rt = reward.copy()
            all_done = done.copy()
        else:
            rt += reward*(1-all_done)
            all_done = np.maximum(done,all_done)
    return rt

def noisy_action(action_softmax,eps=1.,clip=5e-2):
    action_softmax_clipped = np.clip(action_softmax,clip,1-clip)
    logit_unscaled = np.log(action_softmax_clipped)
    u = np.random.rand(*logit_unscaled.shape)
    g = -np.log(-np.log(u))
    return (eps*g+logit_unscaled).argmax(axis=-1)

@click.option('--num_steps', default=20000, type=int)
@click.option('--n_envs', default=10)
@click.option('--n_test_envs', default=10)
@click.option('--n_prev_frames', default=4)
@click.option('--env_id', default='CartPole-v1')
@click.option('--dqda_clipping', default=1.)
@click.option('--clip_norm', default=True, type=bool)
@click.option('--ema_decay', default=0.99, type=float)
@click.option('--hidden_dims', default=32)
@click.option('--hidden_layers', default=2)
@click.option('--output_dim', default=2)
@click.option('--discrete', default=True, type=bool)
@click.option('--add_return_loss', default=False, type=bool)
@click.option('--normalize_inputs', default=True, type=bool)
@click.option('--batchnorm', default=False, type=bool)
@click.option('--add_episode_time_state', default=False, type=bool)
@click.option('--binarized_time_state', default=False, type=bool)
@click.option('--buffer_type', default='normal', type=click.Choice(['normal','prioritized','delayed','delayed_prioritized']))
@click.option('--buffer_size', default=2**11, type=int)
@click.option('--sample_backwards', default=False, type=bool)
@click.option('--prioritized_replay_alpha', default=0.6, type=float)
@click.option('--prioritized_replay_beta0', default=0.4, type=float)
@click.option('--prioritized_replay_beta_iters', default=None, type=int)
@click.option('--prioritized_replay_eps', default=1e-6, type=float)
@click.option('--prioritized_replay_weights_uniform', default=False, type=bool)
@click.option('--prioritized_replay_compute_init', default=False, type=bool)
@click.option('--prioritized_replay_simple', default=False, type=bool)
@click.option('--prioritized_replay_simple_reward_adder', default=4, type=float)
@click.option('--prioritized_replay_simple_done_adder', default=4, type=float)
@click.option('--begin_learning_at_step', default=1e4)
@click.option('--learning_rate', default=1e-4)
@click.option('--learning_rate_decay', default=1.)
@click.option('--learning_rate_q', default=1.)
@click.option('--learning_rate_q_decay', default=1.)
@click.option('--opt_stable_q_online', default=False, type=bool)
@click.option('--opt_stable_q_online_momentum', default=0.99)
@click.option('--stable', default=True, type=bool)
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
    env = VecGymEnv(cfg['env_id'],n_envs=cfg['n_envs'])
    env = NPrevFramesStateEnv(env,n_prev_frames=cfg['n_prev_frames'],flatten=True)

    test_env = VecGymEnv(cfg['env_id'],n_envs=cfg['n_test_envs'])
    test_env = NPrevFramesStateEnv(test_env,n_prev_frames=cfg['n_prev_frames'],flatten=True)

    if cfg['add_episode_time_state']:
        print('ADDING EPISODE TIME STATES')
        env = AddEpisodeTimeStateEnv(env,binarized=cfg['binarized_time_state']) 
        test_env = AddEpisodeTimeStateEnv(test_env,binarized=cfg['binarized_time_state']) 

    state = env.reset()
    state_shape = state.shape
    print('STATE SHAPE: ',state_shape)
    action_shape = env.env.action_shape()

    # Agent
    policy_fn = build_policy_fn(
            cfg['hidden_dims'],
            cfg['hidden_layers'],
            cfg['output_dim'],
            cfg['batchnorm'],
            cfg['normalize_inputs']
    )

    q_fn = build_q_fn(
            cfg['hidden_dims'],
            cfg['hidden_layers'],
            1,
            cfg['batchnorm'],
            cfg['normalize_inputs']
    )

    optimizer_q_kwargs = {}
    if cfg['optimizer_q_decay'] is not None:
        optimizer_q_kwargs['decay'] = cfg['optimizer_q_decay']
    if cfg['optimizer_q_momentum'] is not None:
        optimizer_q_kwargs['momentum'] = cfg['optimizer_q_momentum']
    if cfg['optimizer_q_use_nesterov'] is not None:
        optimizer_q_kwargs['use_nesterov'] = cfg['optimizer_q_use_nesterov']
    agent = StableDDPG(
        state_shape[1:],[2],policy_fn,q_fn,
        cfg['dqda_clipping'],cfg['clip_norm'],
        discrete=cfg['discrete'],
        alpha=cfg['alpha'],
        beta=cfg['beta'],
        optimizer_q=cfg['optimizer_q'],
        opt_q_layerwise=cfg['opt_q_layerwise'],
        optimizer_q_kwargs=optimizer_q_kwargs,
        regularize_policy=cfg['regularize_policy'],
        straight_through_estimation=cfg['straight_through_estimation'],
        add_return_loss=cfg['add_return_loss'],
        stable=cfg['stable'],
        opt_stable_q_online=cfg['opt_stable_q_online'],
        opt_stable_q_online_momentum=cfg['opt_stable_q_online_momentum'],
    )

    # Replay Buffer
    if cfg['buffer_type'] == 'prioritized':
        replay_buffer = PrioritizedBufferMap(
                cfg['buffer_size'],
                alpha=cfg['prioritized_replay_alpha'],
                eps=cfg['prioritized_replay_eps'])
        assert cfg['add_return_loss']==False
    elif cfg['buffer_type'] == 'delayed_prioritized':
        replay_buffer = DelayedPrioritizedBufferMap(
                max_length=cfg['buffer_size'],
                alpha=cfg['prioritized_replay_alpha'],
                eps=cfg['prioritized_replay_eps'],
                add_return_loss=cfg['add_return_loss'],
                )
    elif cfg['buffer_type'] == 'delayed':
        replay_buffer = DelayedBufferMap(cfg['buffer_size'],add_return_loss=cfg['add_return_loss'])
    else:
        replay_buffer = BufferMap(cfg['buffer_size'])
        assert cfg['add_return_loss']==False

    log = LogsTFSummary(savedir)

    state = env.reset()

    start_time = time.time()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())

        T = cfg['num_steps']
        T_beta = T if cfg['prioritized_replay_beta_iters'] is None else cfg['prioritized_replay_beta_iters']
        beta0 = cfg['prioritized_replay_beta0']
        pb = tf.keras.utils.Progbar(T,stateful_metrics=['test_ep_returns','avg_action'])
        for t in range(T):
            start_step_time = time.time()

            action = agent.act(state)

            #if len(replay_buffer) < cfg['begin_learning_at_step'] or np.random.rand() <= 0.02:
            #    action = np.random.choice(action.shape[1],size=len(action))
            #else:
            #    action = action.argmax(axis=1)
            if len(replay_buffer) >= cfg['begin_learning_at_step']:
                action = noisy_action(action)
            else:
                action = np.random.choice(action.shape[1],size=len(action))

            state2, reward, done, info = env.step(action.astype('int').ravel())

            data = {
                'state':state,
                'action':onehot(action),
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

                if t >= cfg['begin_learning_at_step'] + cfg['n_steps_train_only_q']:
                    t2 = t - (cfg['begin_learning_at_step'] + cfg['n_steps_train_only_q'])
                    policy_learning_rate = cfg['learning_rate']*(cfg['learning_rate_decay']**t2)
                else:
                    policy_learning_rate = 0.

                tq = t - cfg['begin_learning_at_step']
                learning_rate_q = cfg['learning_rate_q']*(cfg['learning_rate_q_decay']**tq)


                for i in range(cfg['n_update_steps']):
                    if cfg['buffer_type'] == 'prioritized':

                        beta = beta0 + (1.-beta0)*min(1.,float(t)/T_beta)
                        import ipdb
                        log.append('beta',beta)

                        if cfg['sample_backwards']:
                            raise NotImplementedError("sample_backwards not implemented for prioritized buffer")
                        else:
                            sample = replay_buffer.sample(cfg['batchsize'],beta=beta)
                        if cfg['prioritized_replay_weights_uniform']:
                            sample['importance_weight'] = np.ones_like(sample['importance_weight'])
                        log.append('max_importance_weight',sample['importance_weight'].max())

                        td_error, Q_ema_state2 = agent.update(
                                learning_rate=policy_learning_rate,
                                learning_rate_q=learning_rate_q,
                                ema_decay=cfg['ema_decay'],
                                gamma=cfg['gamma'],
                                weight_decay=cfg['weight_decay'],
                                entropy_loss_weight=cfg['entropy_loss_weight'],
                                outputs=['td_error','Q_ema_state2'],
                                **sample)

                        if not cfg['prioritized_replay_simple']:
                            replay_buffer.update_priorities(td_error)
                    else:

                        if cfg['sample_backwards']:
                            sample = replay_buffer.sample_backwards(cfg['batchsize'])
                        else:
                            sample = replay_buffer.sample(cfg['batchsize'])

                        Q_ema_state2, = agent.update(
                                learning_rate=policy_learning_rate,
                                learning_rate_q=learning_rate_q,
                                ema_decay=cfg['ema_decay'],
                                gamma=cfg['gamma'],
                                weight_decay=cfg['weight_decay'],
                                entropy_loss_weight=cfg['entropy_loss_weight'],
                                outputs=['Q_ema_state2'],
                                **sample)
                pb_input.append(('Q_ema_state2', Q_ema_state2.mean()))
                log.append('Q',Q_ema_state2.mean())

            # Bookkeeping
            log.append('reward_history',reward)
            log.append('action_history',action)
            avg_action = np.mean(log['action_history'][-20:])
            pb_input.append(('avg_action', avg_action))
            if t % cfg['n_steps_per_eval'] == 0 and t > 0:
                log.append('test_ep_returns',test_agent(test_env,agent))
                log.append('test_ep_steps',t)
                #log.append('test_duration_cumulative',time.time()-start_time)
                avg_test_ep_returns = np.mean(log['test_ep_returns'][-1:])
                pb_input.append(('test_ep_returns', avg_test_ep_returns))
            end_time = time.time()
            log.append('step_duration_sec',end_time-start_step_time)
            log.append('duration_cumulative',end_time-start_time)

            pb.add(1,pb_input)
            log.flush(step=t)

    if cfg['buffer_type'] == 'prioritized':
        log.append('priorities', replay_buffer.priorities())
        log.append('sample_importance_weights', replay_buffer.sample(100,beta=1.)['importance_weight'])

    log.write(os.path.join(savedir,'log.h5'))

if __name__=='__main__':
    click.command()(run)()
