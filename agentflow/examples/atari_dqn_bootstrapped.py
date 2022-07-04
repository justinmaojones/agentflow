import click
import gc
import h5py
import numpy as np
import os
import tensorflow as tf
import time
import yaml 

from agentflow.agents import BootstrappedDQN
from agentflow.buffers import BootstrapMaskBuffer
from agentflow.buffers import BufferMap
from agentflow.buffers import CompressedImageBuffer 
from agentflow.buffers import PrioritizedBufferMap
from agentflow.buffers import NStepReturnBuffer
from agentflow.examples.env import dqn_atari_paper_env
from agentflow.examples.nn import dqn_atari_paper_net
from agentflow.examples.nn import dqn_atari_paper_net_dueling
from agentflow.numpy.ops import onehot
from agentflow.numpy.ops import eps_greedy_noise
from agentflow.numpy.ops import gumbel_softmax_noise
from agentflow.numpy.schedules import ExponentialDecaySchedule 
from agentflow.numpy.schedules import LinearAnnealingSchedule
from agentflow.state import NPrevFramesStateEnv
from agentflow.state import PrevEpisodeReturnsEnv 
from agentflow.state import PrevEpisodeLengthsEnv 
from agentflow.state import RandomOneHotMaskEnv 
from agentflow.state import TestAgentEnv 
from agentflow.tensorflow.nn import dense_net
from agentflow.tensorflow.nn import normalize_ema
from agentflow.utils import LogsTFSummary


@click.option('--env_id', default='PongNoFrameskip-v4', type=str)
@click.option('--frames_per_action', default=1, type=int)
@click.option('--num_steps', default=30000, type=int)
@click.option('--num_frames_max', default=None, type=int)
@click.option('--n_envs', default=1)
@click.option('--n_prev_frames', default=4)
@click.option('--ema_decay', default=0.95, type=float)
@click.option('--target_network_copy_freq', default=None, type=int)
@click.option('--noise_type', default='eps_greedy', type=click.Choice(['eps_greedy', 'gumbel_softmax']))
@click.option('--noise_scale_anneal_steps', default=int(1e6), type=int)
@click.option('--noise_scale_init', default=1.0, type=float)
@click.option('--noise_scale_final', default=0.01, type=float)
@click.option('--noise_temperature', default=1.0, type=float)
@click.option('--dueling', default=False, type=bool)
@click.option('--double_q', default=True, type=bool)
@click.option('--bootstrap_num_heads', default=16, type=int)
@click.option('--bootstrap_mask_prob', default=0.5, type=float)
@click.option('--bootstrap_prior_scale', default=1.0, type=float)
@click.option('--bootstrap_random_prior', default=False, type=bool)
@click.option('--bootstrap_explore_before_learning', default=False, type=bool)
@click.option('--network_scale', default=1, type=float)
@click.option('--batchnorm', default=False, type=bool)
@click.option('--buffer_type', default='normal', type=click.Choice(['normal', 'prioritized', 'delayed', 'delayed_prioritized']))
@click.option('--buffer_size', default=30000, type=int)
@click.option('--encode_obs', default=True, type=bool)
@click.option('--n_step_return', default=8, type=int)
@click.option('--prioritized_replay_alpha', default=0.6, type=float)
@click.option('--prioritized_replay_beta0', default=0.4, type=float)
@click.option('--prioritized_replay_beta_iters', default=None, type=int)
@click.option('--prioritized_replay_eps', default=1e-6, type=float)
@click.option('--prioritized_replay_simple', default=True, type=bool)
@click.option('--prioritized_replay_default_reward_priority', default=5, type=float)
@click.option('--prioritized_replay_default_done_priority', default=5, type=float)
@click.option('--begin_learning_at_step', default=200)
@click.option('--learning_rate', default=1e-4)
@click.option('--learning_rate_final', default=0.0, type=float)
@click.option('--learning_rate_decay', default=0.99995)
@click.option('--td_loss', default='square', type=click.Choice(['square', 'huber']))
@click.option('--grad_clip_norm', default=None, type=float)
@click.option('--normalize_inputs', default=True, type=bool)
@click.option('--gamma', default=0.99)
@click.option('--weight_decay', default=1e-4)
@click.option('--n_update_steps', default=4, type=int)
@click.option('--update_freq', default=1, type=int)
@click.option('--n_steps_per_eval', default=1000, type=int)
@click.option('--batchsize', default=64)
@click.option('--log_flush_freq', default=1000, type=int)
@click.option('--log_pnorms', default=False, type=bool)
@click.option('--savedir', default='results')
@click.option('--seed', default=None, type=int)
def run(**cfg):

    for k in sorted(cfg):
        print('CONFIG: ', k, str(cfg[k]))

    if cfg['seed'] is not None:
        np.random.seed(cfg['seed'])
        tf.random.set_seed(int(10*cfg['seed']))

     
    ts = str(int(time.time()*1000))
    suff = str(np.random.choice(np.iinfo(np.int64).max))
    savedir = os.path.join(cfg['savedir'], 'experiment' + ts + '_' + suff)
    print('SAVING TO: {savedir}'.format(**locals()))
    os.system('mkdir -p {savedir}'.format(**locals()))

    with open(os.path.join(savedir, 'config.yaml'), 'w') as f:
        yaml.dump(cfg, f)

    # environment
    env = dqn_atari_paper_env(
        cfg['env_id'], 
        n_envs=cfg['n_envs'], 
        n_prev_frames=cfg['n_prev_frames'], 
        frames_per_action=cfg['frames_per_action']
    )
    env = RandomOneHotMaskEnv(env, dim=cfg['bootstrap_num_heads'])
    env = PrevEpisodeReturnsEnv(env)
    env = PrevEpisodeLengthsEnv(env)
    test_env = dqn_atari_paper_env(
        cfg['env_id'], 
        n_envs=1, 
        n_prev_frames=cfg['n_prev_frames'], 
        frames_per_action=cfg['frames_per_action']
    )
    test_env = TestAgentEnv(test_env)

    # state and action shapes
    state = env.reset()['state']
    state_shape = state.shape
    action_shape = env.n_actions()
    print('STATE SHAPE: ', state_shape)
    print('ACTION SHAPE: ', action_shape)

    log = LogsTFSummary(savedir)

    # build agent
    def q_fn(state, name=None, **kwargs):
        state = state/255.
        if cfg['normalize_inputs']:
            state = normalize_ema(state, name=name)
        if cfg['dueling']:
            return dqn_atari_paper_net_dueling(
                state,
                n_actions=action_shape,
                n_heads=cfg['bootstrap_num_heads'],
                scale=cfg['network_scale'],
                name=name
            )
        else:
            return dqn_atari_paper_net(
                state,
                n_actions=action_shape,
                n_heads=cfg['bootstrap_num_heads'],
                scale=cfg['network_scale'],
                name=name
            )

    learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate = cfg['learning_rate'],
        decay_rate = cfg['learning_rate_decay'],
        decay_steps = cfg['n_update_steps']
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate_schedule,
    )

    agent = BootstrappedDQN(
        state_shape=state_shape[1:],
        num_actions=action_shape,
        q_fn=q_fn,
        optimizer=optimizer,
        double_q=cfg['double_q'],
        num_heads=cfg['bootstrap_num_heads'],
        random_prior=cfg['bootstrap_random_prior'],
        prior_scale=cfg['bootstrap_prior_scale'],
    )

    for v in agent.weights + agent.weights_target:
        print(v.name, v.shape)

    # Replay Buffer
    if cfg['buffer_type'] == 'prioritized':
        # prioritized experience replay
        replay_buffer = PrioritizedBufferMap(
            max_length = cfg['buffer_size'] // cfg['n_envs'],
            alpha = cfg['prioritized_replay_alpha'],
            eps = cfg['prioritized_replay_eps'],
            default_priority = 1.0,
            default_non_zero_reward_priority = cfg['prioritized_replay_default_reward_priority'],
            default_done_priority = cfg['prioritized_replay_default_done_priority'],
            priority_key = 'priority' if not cfg['prioritized_replay_simple'] else None
        )

        beta_schedule = LinearAnnealingSchedule(
            initial_value = cfg['prioritized_replay_beta0'],
            final_value = 1.0,
            annealing_steps = cfg['prioritized_replay_beta_iters'] or cfg['num_steps'],
            begin_at_step = cfg['begin_learning_at_step'],
        )

    else:
        # Normal Buffer
        replay_buffer = BufferMap(cfg['buffer_size'] // cfg['n_envs'])

    if cfg['encode_obs']:
        replay_buffer = CompressedImageBuffer(
            replay_buffer, 
            max_encoding_size=10000,
            keys_to_encode = ['state', 'state2']
        )

    # Delays publishing of records to the underlying replay buffer for n steps
    # then publishes the discounted n-step return
    if cfg['n_step_return'] > 1:
        replay_buffer = NStepReturnBuffer(
            replay_buffer,
            n_steps=cfg['n_step_return'],
            gamma=cfg['gamma'],
        )

    replay_buffer = BootstrapMaskBuffer(
        replay_buffer,
        depth = cfg['bootstrap_num_heads'],
        sample_prob = cfg['bootstrap_mask_prob']
    )

    # Annealed parameters
    noise_scale_schedule = LinearAnnealingSchedule(
        initial_value = cfg['noise_scale_init'],
        final_value = cfg['noise_scale_final'],
        annealing_steps = cfg['noise_scale_anneal_steps'],
        begin_at_step = 0, 
    )

    log = LogsTFSummary(savedir)

    state_and_mask = env.reset()
    state = state_and_mask['state']
    mask = state_and_mask['mask']

    start_time = time.time()

    T = cfg['num_steps']

    pb = tf.keras.utils.Progbar(T, stateful_metrics=[
        'test_ep_length',
        'test_ep_returns',
        'train_ep_length',
        'train_ep_return',
    ])
    frame_counter = 0
    update_counter = 0
    for t in range(T):
        tf.summary.experimental.set_step(t)

        start_step_time = time.time()

        noise_scale = noise_scale_schedule(t)
        log.append('noise_scale', noise_scale)
        if frame_counter >= cfg['begin_learning_at_step'] or cfg['bootstrap_explore_before_learning']:
            if cfg['noise_type'] == 'eps_greedy':
                logits = agent.policy_logits(state, mask).numpy()
                action = eps_greedy_noise(logits, eps=noise_scale)
            elif cfg['noise_type'] == 'gumbel_softmax':
                logits = agent.policy_logits(state, mask).numpy()
                action = gumbel_softmax_noise(logits, temperature=cfg['noise_temperature'])
            else:
                raise NotImplementedError("unknown noise type %s" % cfg['noise_type'])
        else:
            # completely random action choices
            action = np.random.choice(action_shape, size=cfg['n_envs'])

        step_output = env.step(action.astype('int').ravel())

        data = {
            'state':state,
            'action':onehot(action, depth=action_shape),
            'reward':step_output['reward'],
            'done':step_output['done'],
            'state2':step_output['state'],
        }
        if cfg['buffer_type'] == 'prioritized' and not cfg['prioritized_replay_simple']:
            raise NotImplementedError("infer not implemented")
            data['priority'] = agent.infer(
                outputs=['abs_td_error'], 
                gamma=cfg['gamma'],
                **data
            )['abs_td_error']

        log.append('train/ep_return', step_output['prev_episode_return'])
        log.append('train/ep_length', step_output['prev_episode_length'])
        replay_buffer.append(data)
        state = step_output['state']
        mask = step_output['mask']

        pb_input = [
            ('train_ep_return', step_output['prev_episode_return'].mean()),
            ('train_ep_length', step_output['prev_episode_length'].mean()),
        ]
        if frame_counter >= cfg['begin_learning_at_step'] \
                and t % cfg['update_freq'] == 0 \
                and len(replay_buffer) > 0:

            for i in range(cfg['n_update_steps']):
                if cfg['buffer_type'] == 'prioritized':
                    beta = beta_schedule(t)
                    sample = replay_buffer.sample(cfg['batchsize'], beta=beta)
                    log.append('beta', beta)
                    log.append('max_importance_weight', sample['importance_weight'].max())
                else:
                    sample = replay_buffer.sample(cfg['batchsize'])

                ema_decay = cfg['ema_decay']
                if cfg['target_network_copy_freq'] is not None:
                    if t % cfg['target_network_copy_freq'] == 0 and i == 0:
                        ema_decay = 0.0
                log.append('ema_decay', ema_decay)

                update_outputs = agent.update(
                        ema_decay=ema_decay,
                        gamma=cfg['gamma'],
                        weight_decay=cfg['weight_decay'],
                        grad_clip_norm=cfg['grad_clip_norm'],
                        outputs=['td_error', 'Q_policy_eval', 'loss'],
                        **sample)
                update_counter += 1

                if cfg['buffer_type'] == 'prioritized' and not cfg['prioritized_replay_simple']:
                    replay_buffer.update_priorities(update_outputs['abs_td_error'])

            log.append('learning_rate', agent.learning_rate)
            log.append_dict(update_outputs)

            if cfg['log_pnorms']:
                log.append_dict(agent.pnorms())

        if t % cfg['n_steps_per_eval'] == 0 and t >= cfg['begin_learning_at_step']:
            test_output = test_env.test(agent)
            log.append('test_ep_returns', test_output['return'])
            log.append('test_ep_length', test_output['length'])
            log.append('test_ep_t', t)
            log.append('test_ep_steps', frame_counter)
            avg_test_ep_length = np.mean(log['test_ep_length'][-1:])
            avg_test_ep_returns = np.mean(log['test_ep_returns'][-1:])
            pb_input.append(('test_ep_length', avg_test_ep_length))
            pb_input.append(('test_ep_returns', avg_test_ep_returns))

        end_time = time.time()
        frame_counter += cfg['n_envs']
        log.append('step', t+1)
        log.append('frame', frame_counter)
        log.append('update', update_counter)
        log.append('step_duration_sec', end_time-start_step_time)
        log.append('duration_cumulative', end_time-start_time)

        pb.add(1, pb_input)

        if t % cfg['log_flush_freq'] == 0 and t > 0:
            log.flush()
            gc.collect()

        if cfg['num_frames_max'] is not None and frame_counter >= cfg['num_frames_max']:
            print("Stopping program because frame_counter=%0.0f has exceeded num_frames_max=%0.0f" % (frame_counter, cfg['num_frames_max']))
            break

    log.write(os.path.join(savedir, 'log.h5'))

if __name__=='__main__':
    click.command()(run)()
