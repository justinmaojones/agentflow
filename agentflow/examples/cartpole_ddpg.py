import click
import h5py
import numpy as np
import os
import tensorflow as tf
import time
import yaml 

from agentflow.env import VecGymEnv
from agentflow.agents import DiscreteDDPG
from agentflow.agents.utils import test_agent
from agentflow.buffers import BufferMap
from agentflow.buffers import PrioritizedBufferMap
from agentflow.buffers import NStepReturnPublisher
from agentflow.numpy.ops import onehot
from agentflow.numpy.ops import eps_greedy_noise
from agentflow.numpy.ops import gumbel_softmax_noise
from agentflow.numpy.schedules import ExponentialDecaySchedule 
from agentflow.numpy.schedules import LinearAnnealingSchedule
from agentflow.state import NPrevFramesStateEnv
from agentflow.tensorflow.nn import dense_net
from agentflow.tensorflow.ops import normalize_ema
from agentflow.utils import LogsTFSummary

@click.option('--num_steps', default=30000, type=int)
@click.option('--n_envs', default=1)
@click.option('--n_prev_frames', default=16)
@click.option('--dqda_clipping', default=1.)
@click.option('--clip_norm', default=True, type=bool)
@click.option('--ema_decay', default=0.95, type=float)
@click.option('--noise', default='gumbel_softmax', type=click.Choice(['eps_greedy','gumbel_softmax']))
@click.option('--noise_eps', default=0.05, type=float)
@click.option('--noise_temperature', default=0.5, type=float)
@click.option('--hidden_dims', default=64)
@click.option('--hidden_layers', default=4)
@click.option('--policy_temp', default=1.0, type=float)
@click.option('--q_temp', default=1.0, type=float)
@click.option('--noisy_target', default=0.0, type=float)
@click.option('--normalize_inputs', default=True, type=bool)
@click.option('--batchnorm', default=False, type=bool)
@click.option('--binarized_time_state', default=False, type=bool)
@click.option('--buffer_type', default='prioritized', type=click.Choice(['normal','prioritized','delayed','delayed_prioritized']))
@click.option('--buffer_size', default=30000, type=int)
@click.option('--sample_backwards', default=False, type=bool)
@click.option('--enable_n_step_return_publisher', default=True, type=bool)
@click.option('--n_step_return', default=4, type=int)
@click.option('--prioritized_replay_alpha', default=0.25, type=float)
@click.option('--prioritized_replay_beta0', default=0.5, type=float)
@click.option('--prioritized_replay_beta_iters', default=10000, type=int)
@click.option('--prioritized_replay_eps', default=1e-6, type=float)
@click.option('--prioritized_replay_simple', default=False, type=bool)
@click.option('--prioritized_replay_default_reward_priority', default=5, type=float)
@click.option('--prioritized_replay_default_done_priority', default=5, type=float)
@click.option('--begin_learning_at_step', default=200)
@click.option('--learning_rate', default=0.005)
@click.option('--learning_rate_decay', default=0.999)
@click.option('--policy_loss_weight', default=2.0)
@click.option('--gamma', default=0.99)
@click.option('--weight_decay', default=1e-3)
@click.option('--regularize_policy', default=False, type=bool)
@click.option('--entropy_loss_weight', default=1e-4)
@click.option('--entropy_loss_weight_decay', default=0.99995)
@click.option('--n_update_steps', default=4, type=int)
@click.option('--update_freq', default=1, type=int)
@click.option('--n_steps_per_eval', default=100, type=int)
@click.option('--batchsize', default=64)
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

    # environment
    env = VecGymEnv('CartPole-v1', n_envs=cfg['n_envs'])
    env = NPrevFramesStateEnv(env, n_prev_frames=cfg['n_prev_frames'], flatten=True)
    test_env = VecGymEnv('CartPole-v1', n_envs=1)
    test_env = NPrevFramesStateEnv(test_env, n_prev_frames=cfg['n_prev_frames'], flatten=True)

    # state and action shapes
    state = env.reset()['state']
    state_shape = state.shape
    print('STATE SHAPE: ', state_shape)

    action_shape = 2
    print('ACTION SHAPE: ', action_shape)

    # build agent
    def policy_fn(state, training=False):
        if cfg['normalize_inputs']:
            state, _ = normalize_ema(state, training)
        h = dense_net(
            state, 
            cfg['hidden_dims'],
            cfg['hidden_layers'],
            batchnorm = cfg['batchnorm'],
            training = training
        )
        logits = tf.layers.dense(h, action_shape)
        return logits

    def q_fn(state, action, training=False, **kwargs):
        if cfg['normalize_inputs']:
            state, _ = normalize_ema(state, training)
        h = dense_net(
            tf.concat([state,action],axis=1),
            cfg['hidden_dims'],
            cfg['hidden_layers'],
            batchnorm = cfg['batchnorm'],
            training = training
        )
        return tf.layers.dense(h,1)

    agent = DiscreteDDPG(
        state_shape=state_shape[1:],
        num_actions=action_shape,
        policy_fn=policy_fn,
        q_fn=q_fn,
        dqda_clipping=cfg['dqda_clipping'],
        clip_norm=cfg['clip_norm'],
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
        replay_buffer = NStepReturnPublisher(
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

    entropy_loss_weight_schedule = ExponentialDecaySchedule(
        initial_value = cfg['entropy_loss_weight'],
        final_value = 0.0,
        decay_rate = cfg['learning_rate_decay'],
        begin_at_step = cfg['begin_learning_at_step']
    )

    log = LogsTFSummary(savedir)

    state = env.reset()['state']

    start_time = time.time()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())

        T = cfg['num_steps']

        pb = tf.keras.utils.Progbar(T,stateful_metrics=['test_ep_returns'])
        for t in range(T):
            start_step_time = time.time()

            action_probs = agent.act_probs(state)

            if len(replay_buffer) >= cfg['begin_learning_at_step']:
                if cfg['noise'] == 'eps_greedy':
                    action = eps_greedy_noise(action_probs, eps=cfg['noise_eps'])
                elif cfg['noise'] == 'gumbel_softmax':
                    action = gumbel_softmax_noise(action_probs, temperature=cfg['noise_temperature'])
                else:
                    raise NotImplementedError("unknown noise type %s" % cfg['noise'])
            else:
                # completely random action choices
                action = eps_greedy_noise(action_probs, eps=1.0)

            step_output = env.step(action.astype('int').ravel())

            data = {
                'state':state,
                'action':onehot(action),
                'reward':step_output['reward'],
                'done':step_output['done'],
                'state2':step_output['state'],
            }
            log.append_dict(data)
            replay_buffer.append(data)
            state = data['state2']

            pb_input = []
            if t >= cfg['begin_learning_at_step'] and t % cfg['update_freq'] == 0:

                learning_rate = learning_rate_schedule(t)
                entropy_loss_weight = entropy_loss_weight_schedule(t)

                for i in range(cfg['n_update_steps']):
                    if cfg['buffer_type'] == 'prioritized':
                        beta = beta_schedule(t)
                        sample = replay_buffer.sample(cfg['batchsize'],beta=beta)
                        log.append('beta',beta)
                        log.append('max_importance_weight',sample['importance_weight'].max())
                    else:
                        sample = replay_buffer.sample(cfg['batchsize'])

                    update_outputs = agent.update(
                            learning_rate=learning_rate,
                            ema_decay=cfg['ema_decay'],
                            gamma=cfg['gamma'],
                            weight_decay=cfg['weight_decay'],
                            policy_loss_weight=cfg['policy_loss_weight'],
                            entropy_loss_weight=entropy_loss_weight,
                            outputs=['td_error','Q_policy_eval'],
                            **sample)

                    if cfg['buffer_type'] == 'prioritized' and not cfg['prioritized_replay_simple']:
                        replay_buffer.update_priorities(update_outputs['td_error'])

                log.append_dict(update_outputs)

            if t % cfg['n_steps_per_eval'] == 0 and t > 0:
                log.append('test_ep_returns',test_agent(test_env,agent))
                log.append('test_ep_steps',t)
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
