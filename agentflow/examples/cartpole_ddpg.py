from agentflow.env import VecGymEnv
from agentflow.agents import DDPG
from agentflow.buffers import BufferMap, DelayedBufferMap, PrioritizedBufferMap, DelayedPrioritizedBufferMap, NStepReturnPublisher
from agentflow.state import NPrevFramesStateEnv, AddEpisodeTimeStateEnv
from agentflow.numpy.ops import onehot
from agentflow.numpy.schedules import LinearAnnealingSchedule
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

def build_net_fn(hidden_dims,hidden_layers,output_dim,batchnorm,dropout=0.0):
    def net_fn(h,training=False):
        h = dense_net(h,hidden_dims,hidden_layers,batchnorm=batchnorm,training=training,dropout=dropout)
        return tf.layers.dense(h,output_dim)
    return net_fn

def build_policy_fn(hidden_dims,hidden_layers,output_dim,batchnorm,normalize_inputs=True,temp_eval=1.0):
    net_fn = build_net_fn(hidden_dims,hidden_layers,output_dim,batchnorm)
    def policy_fn(state,training=False):
        if normalize_inputs:
            state, _ = normalize_ema(state,training)
        logits = net_fn(state,training)
        if not training:
            logits *= temp_eval
        return tf.nn.softmax(logits,axis=-1), logits, state 
    return policy_fn

def build_q_fn(hidden_dims,hidden_layers,output_dim,batchnorm,normalize_inputs=True,dropout=0.0):
    net_fn = build_net_fn(hidden_dims,hidden_layers,output_dim,batchnorm,dropout)
    def q_fn(state,action,training=False,q_normalized=False,**kwargs):
        # q_normalized is a dummy kwarg
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
@click.option('--dropout', default=0.0, type=float)
@click.option('--hidden_dims', default=32)
@click.option('--hidden_layers', default=2)
@click.option('--output_dim', default=2)
@click.option('--policy_temp', default=1.0, type=float)
@click.option('--q_temp', default=1.0, type=float)
@click.option('--noisy_target', default=0.0, type=float)
@click.option('--normalize_inputs', default=True, type=bool)
@click.option('--batchnorm', default=False, type=bool)
@click.option('--binarized_time_state', default=False, type=bool)
@click.option('--buffer_type', default='normal', type=click.Choice(['normal','prioritized','delayed','delayed_prioritized']))
@click.option('--buffer_size', default=2**11, type=int)
@click.option('--sample_backwards', default=False, type=bool)
@click.option('--enable_n_step_return_publisher', default=False, type=bool)
@click.option('--n_step_return', default=1, type=int)
@click.option('--prioritized_replay_alpha', default=0.6, type=float)
@click.option('--prioritized_replay_beta0', default=0.4, type=float)
@click.option('--prioritized_replay_beta_iters', default=None, type=int)
@click.option('--prioritized_replay_eps', default=1e-6, type=float)
@click.option('--prioritized_replay_simple', default=False, type=bool)
@click.option('--prioritized_replay_default_reward_priority', default=5, type=float)
@click.option('--prioritized_replay_default_done_priority', default=5, type=float)
@click.option('--begin_learning_at_step', default=1e4)
@click.option('--learning_rate', default=1e-4)
@click.option('--learning_rate_decay', default=1.)
@click.option('--policy_loss_weight', default=1e-3)
@click.option('--gamma', default=0.99)
@click.option('--weight_decay', default=0.0)
@click.option('--regularize_policy', default=False, type=bool)
@click.option('--straight_through_estimation', default=False, type=bool)
@click.option('--entropy_loss_weight', default=0.0)
@click.option('--entropy_loss_weight_decay', default=0.99995)
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

    state = env.reset()
    state_shape = state.shape
    print('STATE SHAPE: ',state_shape)

    # Agent
    policy_fn = build_policy_fn(
        cfg['hidden_dims'],
        cfg['hidden_layers'],
        cfg['output_dim'],
        cfg['batchnorm'],
        cfg['normalize_inputs'],
        temp_eval=cfg['policy_temp'],
    )

    q_fn = build_q_fn(
        cfg['hidden_dims'],
        cfg['hidden_layers'],
        1,
        cfg['batchnorm'],
        cfg['normalize_inputs'],
        dropout=cfg['dropout'],
    )

    agent = DDPG(
        state_shape=state_shape[1:],
        action_shape=[2],
        policy_fn=policy_fn,
        q_fn=q_fn,
        dqda_clipping=cfg['dqda_clipping'],
        clip_norm=cfg['clip_norm'],
        discrete=True,
    )

    # Replay Buffer
    if cfg['buffer_type'] == 'prioritized':
        # Prioritized Experience Replay
        replay_buffer = PrioritizedBufferMap(
            max_length = cfg['buffer_size'],
            alpha = cfg['prioritized_replay_alpha'],
            eps = cfg['prioritized_replay_eps'],
            default_priority = 1.0,
            default_non_zero_reward_priority = cfg['prioritized_replay_default_reward_priority'],
            default_done_priority = cfg['prioritized_replay_default_done_priority'],
        )

        beta_schedule = LinearAnnealingSchedule(
            initial_value = cfg['prioritized_replay_beta0'],
            final_value = 1.0,
            annealing_steps = cfg['prioritized_replay_beta_iters'] or cfg['num_steps'],
            begin_at_step = cfg['begin_learning_at_step'],
        )

    else:
        # Normal Buffer
        replay_buffer = BufferMap(cfg['buffer_size'])

    # Delays publishing of records to the underlying replay buffer for n steps
    # then publishes the discounted n-step return
    if cfg['enable_n_step_return_publisher']:
        replay_buffer = NStepReturnPublisher(
            replay_buffer,
            n_steps=cfg['n_step_return'],
            gamma=cfg['gamma'],
        )

    log = LogsTFSummary(savedir)

    state = env.reset()

    start_time = time.time()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())

        T = cfg['num_steps']

        pb = tf.keras.utils.Progbar(T,stateful_metrics=['test_ep_returns','avg_action'])
        for t in range(T):
            start_step_time = time.time()

            action = agent.act(state)

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
            replay_buffer.append(data)
            state = state2 

            pb_input = []
            if t >= cfg['begin_learning_at_step'] and t % cfg['update_freq'] == 0:

                if t >= cfg['begin_learning_at_step']:
                    t2 = t - cfg['begin_learning_at_step']
                    learning_rate = cfg['learning_rate']*(cfg['learning_rate_decay']**t2)
                    entropy_loss_weight = cfg['entropy_loss_weight']*(cfg['entropy_loss_weight_decay']**t2)
                else:
                    learning_rate = 0.
                    entropy_loss_weight = cfg['entropy_loss_weight']

                for i in range(cfg['n_update_steps']):
                    if cfg['buffer_type'] == 'prioritized':

                        beta = beta_schedule(t)
                        log.append('beta',beta)

                        sample = replay_buffer.sample(cfg['batchsize'],beta=beta)
                        log.append('max_importance_weight',sample['importance_weight'].max())

                        update_outputs = agent.update(
                                learning_rate=learning_rate,
                                ema_decay=cfg['ema_decay'],
                                gamma=cfg['gamma'],
                                weight_decay=cfg['weight_decay'],
                                policy_loss_weight=cfg['policy_loss_weight'],
                                entropy_loss_weight=entropy_loss_weight,
                                outputs=['td_error','Q_ema_state2'],
                                **sample)

                        if not cfg['prioritized_replay_simple']:
                            replay_buffer.update_priorities(update_outputs['td_error'])
                    else:

                        sample = replay_buffer.sample(cfg['batchsize'])

                        update_outputs = agent.update(
                                learning_rate=learning_rate,
                                ema_decay=cfg['ema_decay'],
                                gamma=cfg['gamma'],
                                weight_decay=cfg['weight_decay'],
                                policy_loss_weight=cfg['policy_loss_weight'],
                                entropy_loss_weight=entropy_loss_weight,
                                outputs=['Q_ema_state2'],
                                **sample)
                pb_input.append(('Q_ema_state2', update_outputs['Q_ema_state2'].mean()))
                log.append('Q',update_outputs['Q_ema_state2'].mean())

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

    log.write(os.path.join(savedir,'log.h5'))

if __name__=='__main__':
    click.command()(run)()
