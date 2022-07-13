import click
import h5py
import numpy as np
import os
import tensorflow as tf
import time
import yaml 

from agentflow.env import CartpoleGymEnv
from agentflow.agents import CompletelyRandomDiscreteUntil 
from agentflow.agents import DQN
from agentflow.agents import EpsilonGreedy
from agentflow.buffers import ActionToOneHotBuffer
from agentflow.buffers import PrioritizedBufferMap
from agentflow.buffers import NStepReturnBuffer
from agentflow.logging import remote_scoped_log_tf_summary 
from agentflow.numpy.schedules import ExponentialDecaySchedule 
from agentflow.numpy.schedules import LinearAnnealingSchedule
from agentflow.state import NPrevFramesStateEnv
from agentflow.state import PrevEpisodeReturnsEnv 
from agentflow.state import PrevEpisodeLengthsEnv 
from agentflow.tensorflow.nn import dense_net
from agentflow.tensorflow.nn import normalize_ema
from agentflow.train import AsyncTrainer

@click.option('--num_steps', default=30000, type=int)
@click.option('--n_envs', default=16)
@click.option('--n_prev_frames', default=16)
@click.option('--runner_count', default=1, type=int)
@click.option('--runner_cpu', default=1, type=int)
@click.option('--runner_threads', default=2, type=int)
@click.option('--parameter_server_cpu', default=1, type=int)
@click.option('--parameter_server_gpu', default=0, type=int)
@click.option('--parameter_server_threads', default=2, type=int)
@click.option('--ema_decay', default=0.99, type=float)
@click.option('--noise_eps', default=0.5, type=float)
@click.option('--noise_temperature', default=1.0, type=float)
@click.option('--hidden_dims', default=64)
@click.option('--hidden_layers', default=4)
@click.option('--buffer_size', default=30000, type=int)
@click.option('--n_step_return', default=8, type=int)
@click.option('--prioritized_replay_alpha', default=0.6, type=float)
@click.option('--prioritized_replay_eps', default=1e-6, type=float)
@click.option('--prioritized_replay_default_reward_priority', default=5, type=float)
@click.option('--prioritized_replay_default_done_priority', default=5, type=float)
@click.option('--begin_learning_at_frame', default=200)
@click.option('--n_updates_per_model_refresh', default=32)
@click.option('--learning_rate', default=1e-4)
@click.option('--learning_rate_decay', default=0.9999)
@click.option('--adam_eps', default=1e-7)
@click.option('--gamma', default=0.99)
@click.option('--weight_decay', default=1e-4)
@click.option('--batchsize', default=64)
@click.option('--savedir', default='results/cartpole_dqn_async')
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

    log = remote_scoped_log_tf_summary(savedir)

    # environment
    env = CartpoleGymEnv(n_envs=cfg['n_envs'])
    env = NPrevFramesStateEnv(env, n_prev_frames=cfg['n_prev_frames'], flatten=True)
    env = PrevEpisodeReturnsEnv(env) 
    env = PrevEpisodeLengthsEnv(env)
    test_env = CartpoleGymEnv(n_envs=1)
    test_env = NPrevFramesStateEnv(test_env, n_prev_frames=cfg['n_prev_frames'], flatten=True)

    # state and action shapes
    state = env.reset()['state']
    state_shape = state.shape
    print('STATE SHAPE: ', state_shape)

    num_actions = 2
    print('ACTION SHAPE: ', num_actions)


    # build agent
    def q_fn(state, name=None, **kwargs):
        state = normalize_ema(state, name=name)
        h = dense_net(
            state,
            cfg['hidden_dims'],
            cfg['hidden_layers'],
            name = name
        )
        return tf.keras.layers.Dense(
            num_actions, 
            name=f"{name}/dense/output" if name is not None else "dense/output",
        )(h)

    learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate = cfg['learning_rate'],
        decay_rate = cfg['learning_rate_decay'],
        decay_steps = cfg['n_updates_per_model_refresh']
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate_schedule,
        epsilon=cfg['adam_eps'],
    )

    agent = DQN(
        state_shape=state_shape[1:],
        num_actions=num_actions,
        q_fn=q_fn,
        optimizer=optimizer,
        auto_build=False
    )
    test_agent = agent

    agent = EpsilonGreedy(agent, epsilon=cfg['noise_eps'])
    agent = CompletelyRandomDiscreteUntil(agent, num_steps=cfg['begin_learning_at_frame'])

    # prioritized experience replay
    replay_buffer = PrioritizedBufferMap(
        max_length = cfg['buffer_size'],
        alpha = cfg['prioritized_replay_alpha'],
        eps = cfg['prioritized_replay_eps'],
        default_priority = 1.0,
        default_non_zero_reward_priority = cfg['prioritized_replay_default_reward_priority'],
        default_done_priority = cfg['prioritized_replay_default_done_priority'],
    )

    # Delays publishing of records to the underlying replay buffer for n steps
    # then publishes the discounted n-step return
    replay_buffer = NStepReturnBuffer(
        replay_buffer,
        n_steps=cfg['n_step_return'],
        gamma=cfg['gamma'],
    )

    replay_buffer = ActionToOneHotBuffer(replay_buffer, num_actions)


    print("Build AsyncTrainer")
    trainer = AsyncTrainer(
        env=env, 
        agent=agent, 
        replay_buffer=replay_buffer, 
        begin_learning_at_frame=cfg['begin_learning_at_frame'],
        n_updates_per_model_refresh=cfg['n_updates_per_model_refresh'],
        batchsize=cfg['batchsize'],
        test_env=test_env, 
        test_agent=test_agent, 
        runner_count=cfg['runner_count'],
        runner_cpu=cfg['runner_cpu'],
        runner_threads=cfg['runner_threads'],
        parameter_server_cpu=cfg['parameter_server_cpu'],
        parameter_server_gpu=cfg['parameter_server_gpu'],
        parameter_server_threads=cfg['parameter_server_threads'],
        log=log,
    )

    print("Start learning")
    trainer.learn(cfg['num_steps'])

    log.flush()

if __name__=='__main__':
    click.command()(run)()
