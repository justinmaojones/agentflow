import click
import h5py
import numpy as np
import os
import tensorflow as tf
import time
import yaml 

from agentflow.env import PendulumGymEnv
from agentflow.agents import DDPG
from agentflow.agents.utils import test_agent
from agentflow.buffers import BufferMap
from agentflow.buffers import PrioritizedBufferMap
from agentflow.buffers import NStepReturnBuffer
from agentflow.logging import scoped_log_tf_summary
from agentflow.numpy.ops import onehot
from agentflow.numpy.schedules import ExponentialDecaySchedule 
from agentflow.numpy.schedules import LinearAnnealingSchedule
from agentflow.state import NPrevFramesStateEnv
from agentflow.state import PrevEpisodeReturnsEnv 
from agentflow.state import PrevEpisodeLengthsEnv 
from agentflow.state import TanhActionEnv 
from agentflow.tensorflow.nn import dense_net
from agentflow.tensorflow.nn import normalize_ema


@click.option('--num_steps', default=20000, type=int)
@click.option('--n_envs', default=10)
@click.option('--n_prev_frames', default=4)
@click.option('--dqda_clipping', default=1.)
@click.option('--clip_norm', default=True, type=bool)
@click.option('--ema_decay', default=0.99, type=float)
@click.option('--noise_eps', default=0.05, type=float)
@click.option('--hidden_dims', default=32)
@click.option('--hidden_layers', default=2)
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
@click.option('--grad_clip_norm', default=None)
@click.option('--gamma', default=0.99)
@click.option('--weight_decay', default=0.0)
@click.option('--regularize_policy', default=False, type=bool)
@click.option('--straight_through_estimation', default=False, type=bool)
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
        tf.random.set_seed(int(10*cfg['seed']))

    ts = str(int(time.time()*1000))
    savedir = os.path.join(cfg['savedir'],'experiment' + ts)
    print('SAVING TO: {savedir}'.format(**locals()))
    os.system('mkdir -p {savedir}'.format(**locals()))

    with open(os.path.join(savedir,'config.yaml'),'w') as f:
        yaml.dump(cfg, f)

    # environment
    env = PendulumGymEnv(n_envs=cfg['n_envs'])
    env = NPrevFramesStateEnv(env, n_prev_frames=cfg['n_prev_frames'], flatten=True)
    env = PrevEpisodeReturnsEnv(env)
    env = PrevEpisodeLengthsEnv(env)
    test_env = PendulumGymEnv(n_envs=1)
    test_env = NPrevFramesStateEnv(test_env, n_prev_frames=cfg['n_prev_frames'], flatten=True)

    # state and action shapes
    state = env.reset()['state']
    state_shape = state.shape
    print('STATE SHAPE: ', state_shape)

    action_shape = 1
    print('ACTION SHAPE: ', action_shape)

    log = scoped_log_tf_summary(savedir)

    # build agent
    def policy_fn(state, name=None):
        if cfg['normalize_inputs']:
            state = normalize_ema(state, name=name)
        h = dense_net(
            state, 
            cfg['hidden_dims'],
            cfg['hidden_layers'],
            cfg['batchnorm'],
            name = name
        )
        #return tf.keras.layers.Dense(action_shape, name=f"{name}/dense/output")(h), state
        h = tf.keras.layers.Dense(action_shape, name=f"{name}/dense/output")(h)
        return 2*tf.nn.tanh(h)

    def q_fn(state, action, name=None, **kwargs):
        if cfg['normalize_inputs']:
            state = normalize_ema(state, name=name)
        h = dense_net(
            tf.concat([state,action],axis=1),
            cfg['hidden_dims'],
            cfg['hidden_layers'],
            batchnorm = cfg['batchnorm'],
            name = name
        )
        return tf.keras.layers.Dense(1, name=f"{name}/dense/output")(h)

    learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate = cfg['learning_rate'],
        decay_rate = cfg['learning_rate_decay'],
        decay_steps = cfg['n_update_steps']
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate_schedule,
    )

    agent = DDPG(
        state_shape=state_shape[1:],
        action_shape=[action_shape],
        policy_fn=policy_fn,
        q_fn=q_fn,
        optimizer=optimizer,
        dqda_clipping=cfg['dqda_clipping'],
        clip_norm=cfg['clip_norm'],
        policy_loss_weight=cfg['policy_loss_weight']
    )

    # Replay Buffer
    if cfg['buffer_type'] == 'prioritized':
        # prioritized experience replay
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
        replay_buffer = NStepReturnBuffer(
            replay_buffer,
            n_steps=cfg['n_step_return'],
            gamma=cfg['gamma'],
        )

    # Annealed parameters
    learning_rate_schedule = ExponentialDecaySchedule(
        initial_value = cfg['learning_rate'],
        final_value = 0.0,
        decay_rate = cfg['learning_rate_decay'],
        begin_at_step = cfg['begin_learning_at_step']
    )


    state = env.reset()['state']

    start_time = time.time()

    T = cfg['num_steps']

    pb = tf.keras.utils.Progbar(T,stateful_metrics=['test_ep_returns'])
    for t in range(T):
        tf.summary.experimental.set_step(t)

        start_step_time = time.time()

        if len(replay_buffer) >= cfg['begin_learning_at_step']:
            action = agent.act(state).numpy()
            action += cfg['noise_eps']*np.random.randn(*action.shape)
        else:
            # completely random action choices
            action = np.random.randn(cfg['n_envs'], action_shape)

        step_output = env.step(action)

        data = {
            'state':state,
            'action':action,
            'reward':step_output['reward'],
            'done':step_output['done'],
            'state2':step_output['state'],
        }
        log.append_dict(data)
        log.append('train/ep_return',step_output['episode_return'])
        log.append('train/ep_length',step_output['episode_length'])
        replay_buffer.append(data)
        state = data['state2']

        pb_input = []
        if t >= cfg['begin_learning_at_step'] and t % cfg['update_freq'] == 0:

            for i in range(cfg['n_update_steps']):
                if cfg['buffer_type'] == 'prioritized':
                    beta = beta_schedule(t)
                    sample = replay_buffer.sample(cfg['batchsize'],beta=beta)
                    log.append('beta',beta)
                    log.append('max_importance_weight',sample['importance_weight'].max())
                else:
                    sample = replay_buffer.sample(cfg['batchsize'])

                update_outputs = agent.update(
                        ema_decay=cfg['ema_decay'],
                        gamma=cfg['gamma'],
                        weight_decay=cfg['weight_decay'],
                        grad_clip_norm=cfg['grad_clip_norm'],
                        **sample)

                if cfg['buffer_type'] == 'prioritized' and not cfg['prioritized_replay_simple']:
                    replay_buffer.update_priorities(update_outputs['td_error'])

            log.append('learning_rate', optimizer.learning_rate(optimizer.iterations))
            log.append_dict(update_outputs)

        if t % cfg['n_steps_per_eval'] == 0 and t > 0:
            log.append('test_ep_returns',test_agent(test_env,agent))
            log.append('test_ep_steps',t)
            avg_test_ep_returns = np.mean(log['test_ep_returns'][-1:])
            pb_input.append(('test_ep_returns', avg_test_ep_returns))

        end_time = time.time()
        log.append('step_duration_sec',end_time-start_step_time)
        log.append('duration_cumulative',end_time-start_time)

        pb.add(1, pb_input)
        log.flush()

    log.flush()

if __name__=='__main__':
    click.command()(run)()
