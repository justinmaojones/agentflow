import click
import h5py
import numpy as np
import os
import tensorflow as tf
import time
import yaml 

from agentflow.env import ChainEnv
from agentflow.agents import BootstrappedDQN
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
from agentflow.state import RandomOneHotMaskEnv 
from agentflow.tensorflow.nn import dense_net
from agentflow.tensorflow.ops import normalize_ema
from agentflow.utils import LogsTFSummary

@click.option('--num_steps', default=30000, type=int)
@click.option('--n_envs', default=1)
@click.option('--chain_env_length', default=8)
@click.option('--chain_env_random_action_mask', default=False, type=bool)
@click.option('--ema_decay', default=0.95, type=float)
@click.option('--noise', default='eps_greedy', type=click.Choice(['eps_greedy','gumbel_softmax']))
@click.option('--noise_eps', default=0.05, type=float)
@click.option('--noise_temperature', default=1.0, type=float)
@click.option('--double_q', default=True, type=bool)
@click.option('--bootstrap_num_heads', default=16, type=int)
@click.option('--bootstrap_mask_prob', default=0.5, type=float)
@click.option('--bootstrap_prior_scale', default=0.05, type=float)
@click.option('--bootstrap_random_prior', default=False, type=bool)
@click.option('--hidden_dims', default=64, type=int)
@click.option('--hidden_layers', default=4, type=int)
@click.option('--normalize_inputs', default=True, type=bool)
@click.option('--batchnorm', default=False, type=bool)
@click.option('--buffer_type', default='normal', type=click.Choice(['normal','prioritized','delayed','delayed_prioritized']))
@click.option('--buffer_size', default=30000, type=int)
@click.option('--enable_n_step_return_publisher', default=True, type=bool)
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
@click.option('--learning_rate_decay', default=0.99995)
@click.option('--gamma', default=0.99)
@click.option('--weight_decay', default=1e-4)
@click.option('--entropy_loss_weight', default=1e-5, type=float)
@click.option('--entropy_loss_weight_decay', default=0.99995)
@click.option('--n_update_steps', default=4, type=int)
@click.option('--update_freq', default=1, type=int)
@click.option('--n_steps_per_eval', default=100, type=int)
@click.option('--batchsize', default=64)
@click.option('--log_everything', default=False, type=bool)
@click.option('--savedir', default='results')
@click.option('--seed',default=None, type=int)
def run(**cfg):

    for k in sorted(cfg):
        print('CONFIG: ',k,str(cfg[k]))

    if cfg['seed'] is not None:
        np.random.seed(cfg['seed'])
        tf.set_random_seed(int(10*cfg['seed']))

     
    ts = str(int(time.time()*1000))
    suff = str(np.random.choice(np.iinfo(np.int64).max))
    savedir = os.path.join(cfg['savedir'],'experiment' + ts + '_' + suff)
    print('SAVING TO: {savedir}'.format(**locals()))
    os.system('mkdir -p {savedir}'.format(**locals()))

    with open(os.path.join(savedir,'config.yaml'),'w') as f:
        yaml.dump(cfg, f)

    # environment
    env = ChainEnv(n_envs=cfg['n_envs'], length=cfg['chain_env_length'])
    env = RandomOneHotMaskEnv(env, dim=cfg['bootstrap_num_heads'])
    test_env = env.env.copy(n_envs=1) 
    assert np.abs(env.env.action_mask - test_env.action_mask).max() == 0

    # state and action shapes
    state = env.reset()['state']
    state_shape = state.shape
    print('STATE SHAPE: ', state_shape)

    action_shape = 2
    print('ACTION SHAPE: ', action_shape)

    # build agent
    def q_fn(state, training=False, **kwargs):
        h = dense_net(
            state,
            cfg['hidden_dims'],
            cfg['hidden_layers'],
            batchnorm = cfg['batchnorm'],
            training = training
        )
        output = tf.layers.dense(h,action_shape*cfg['bootstrap_num_heads'])
        return tf.reshape(output,[-1,action_shape,cfg['bootstrap_num_heads']])

    def q_prior_fn(state, training=False, **kwargs):
        h = dense_net(
            state,
            cfg['hidden_dims'],
            cfg['hidden_layers'],
            batchnorm = cfg['batchnorm'],
            training = training,
            kernel_initializer = tf.keras.initializers.VarianceScaling(cfg['bootstrap_prior_scale']),
        )
        output = tf.layers.dense(
            h,
            action_shape*cfg['bootstrap_num_heads'],
            kernel_initializer = tf.keras.initializers.RandomNormal(cfg['bootstrap_prior_scale']),
        )
        return tf.reshape(output,[-1,action_shape,cfg['bootstrap_num_heads']])

    agent = BootstrappedDQN(
        state_shape=state_shape[1:],
        num_actions=action_shape,
        q_fn=q_fn,
        q_prior_fn=q_fn,
        #q_prior_fn=q_prior_fn if cfg['bootstrap_random_prior'] else None,
        double_q=cfg['double_q'],
        num_heads=cfg['bootstrap_num_heads'],
        random_prior=cfg['bootstrap_random_prior'],
        prior_scale=cfg['bootstrap_prior_scale'],
    )

    for v in tf.global_variables():
        print(v.name, v.shape)

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

    state_and_mask = env.reset()
    state = state_and_mask['state']
    mask = state_and_mask['mask']

    start_time = time.time()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())

        T = cfg['num_steps']

        pb = tf.keras.utils.Progbar(T,stateful_metrics=['test_ep_returns'])
        for t in range(T):
            start_step_time = time.time()

            action_probs = agent.act_probs(state, mask)

            if cfg['noise'] == 'eps_greedy':
                action = eps_greedy_noise(action_probs, eps=cfg['noise_eps'])
            elif cfg['noise'] == 'gumbel_softmax':
                action = gumbel_softmax_noise(action_probs, temperature=cfg['noise_temperature'])
            else:
                raise NotImplementedError("unknown noise type %s" % cfg['noise'])

            step_output = env.step(action.astype('int').ravel())
            bootstrap_mask_probs = (1-cfg['bootstrap_mask_prob'],cfg['bootstrap_mask_prob'])
            bootstrap_mask_shape = (len(state),cfg['bootstrap_num_heads'])
            bootstrap_mask = np.random.choice(2, size=bootstrap_mask_shape, p=bootstrap_mask_probs)

            data = {
                'state':state,
                'action':onehot(action),
                'reward':step_output['reward'],
                'done':step_output['done'],
                'state2':step_output['state'],
                'mask':bootstrap_mask,
            }
            replay_buffer.append(data)
            state = step_output['state']
            mask = step_output['mask']

            log.append('reward',data['reward'],summary_only=not cfg['log_everything'])
            log.append('action',data['action'],summary_only=not cfg['log_everything'])
            log.append('done',data['done'],summary_only=not cfg['log_everything'])
            log.append('position',step_output['position'],summary_only=not cfg['log_everything'])
            log.append('time',step_output['time'],summary_only=not cfg['log_everything'])
            log.append('mask',step_output['mask'],summary_only=not cfg['log_everything'])

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

                    if cfg['log_everything']:
                        outputs_to_get = list(agent.outputs.keys())
                    else:
                        outputs_to_get = ['td_error','Q_policy_eval']

                    update_outputs = agent.update(
                            learning_rate=learning_rate,
                            ema_decay=cfg['ema_decay'],
                            gamma=cfg['gamma'],
                            weight_decay=cfg['weight_decay'],
                            entropy_loss_weight=entropy_loss_weight,
                            outputs=outputs_to_get,
                            **sample)

                    if cfg['buffer_type'] == 'prioritized' and not cfg['prioritized_replay_simple']:
                        replay_buffer.update_priorities(update_outputs['td_error'])

                pb_input.append(('Q_policy_eval', update_outputs['Q_policy_eval']))
                log.append_dict(update_outputs)

            if cfg['log_everything']:
                log.append_dict(agent.pnorms())

            if t % cfg['n_steps_per_eval'] == 0 and t > 0:
                log.append('test_ep_returns',test_agent(test_env,agent),summary_only=False)
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
