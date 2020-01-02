from agentflow.env import VecGymEnv
from agentflow.agents import DDPG
from agentflow.buffers import BufferMap, PrioritizedBufferMap
from agentflow.state import NPrevFramesStateEnv
from agentflow.numpy.ops import onehot, noisy_action 
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

def build_net_fn(hidden_dims,hidden_layers,output_dim,batchnorm):
    def net_fn(state,training=False):
        h = state
        if batchnorm:
            BN = tf.layers.BatchNormalization()
            h = BN(h,training=training)
        h = dense_net(h,hidden_dims,hidden_layers,batchnorm=False,training=training)
        return tf.layers.dense(h,output_dim)
    return net_fn

def build_policy_fn(*args,**kwargs):
    net_fn = build_net_fn(*args,**kwargs)
    def policy_fn(state,training=False):
        h = net_fn(state)
        return tf.nn.softmax(h,axis=-1)
    return policy_fn

def build_q_fn(*args,**kwargs):
    net_fn = build_net_fn(*args,**kwargs)
    def q_fn(state,action,training=False):
        h = tf.concat([state,action],axis=1)
        return net_fn(h)
    return q_fn

def test_agent(test_env,agent):
    state, rt, done = test_env.reset(), 0, 0
    while np.sum(done) == 0:
        action = agent.act(state).argmax(axis=-1).ravel()
        state, reward, done, _ = test_env.step(action)
        rt += reward.sum()
    return rt

@click.command()
@click.option('--num_steps', default=20000, type=int)
@click.option('--n_envs', default=10)
@click.option('--env_id', default='CartPole-v1')
@click.option('--dqda_clipping', default=1.)
@click.option('--clip_norm', default=True, type=bool)
@click.option('--hidden_dims', default=64)
@click.option('--hidden_layers', default=3)
@click.option('--output_dim', default=2)
@click.option('--batchnorm', default=True, type=bool)
@click.option('--buffer_type', default='normal', type=click.Choice(['normal','prioritized']))
@click.option('--buffer_size', default=2**11, type=int)
@click.option('--begin_learning_at_step', default=1000)
@click.option('--learning_rate', default=1e-4)
@click.option('--n_update_steps', default=2, type=int)
@click.option('--update_freq', default=4, type=int)
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
        tf.set_random_seed(int(np.random.choice(int(1e6))+1))

    cfg_hash = str(hash(str(sorted(cfg))))
    savedir = os.path.join(cfg['savedir'],'experiment' + cfg_hash)
    os.system('mkdir -p {savedir}'.format(**locals()))

    with open(os.path.join(savedir,'config.yaml'),'w') as f:
        yaml.dump(cfg, f)

    # Environment
    env = VecGymEnv(cfg['env_id'],n_envs=cfg['n_envs'])
    env = NPrevFramesStateEnv(env,n_prev_frames=4,flatten=True)

    test_env = VecGymEnv(cfg['env_id'],n_envs=1)
    test_env = NPrevFramesStateEnv(test_env,n_prev_frames=4,flatten=True)

    state = env.reset()
    state_shape = state.shape
    action_shape = env.env.action_shape()

    # Agent
    policy_fn = build_policy_fn(
            cfg['hidden_dims'],
            cfg['hidden_layers'],
            cfg['output_dim'],
            cfg['batchnorm'])

    q_fn = build_q_fn(
            cfg['hidden_dims'],
            cfg['hidden_layers'],
            1,
            cfg['batchnorm'])

    agent = DDPG(state_shape[1:],[2],policy_fn,q_fn,cfg['dqda_clipping'],cfg['clip_norm'],discrete=discrete)

    # Replay Buffer
    if cfg['buffer_type'] == 'prioritized':
        n_beta_annealing_steps = cfg['num_steps']/cfg['update_freq']*cfg['n_update_steps']
        print('CONFIG: ','n_beta_annealing_steps',n_beta_annealing_steps)
        replay_buffer = PrioritizedBufferMap(
                cfg['buffer_size'],
                alpha=1.,
                eps=0.1,
                n_beta_annealing_steps=n_beta_annealing_steps)
    else:
        replay_buffer = BufferMap(cfg['buffer_size'])

    log = {
        'reward_history': [],
        'action_history': [],
        'test_ep_returns': [],
        'test_ep_steps': [],
        'step_duration_sec': [],
        'duration_cumulative': [],
    }

    state = env.reset()

    start_time = time.time()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        T = cfg['num_steps']
        pb = tf.keras.utils.Progbar(T)
        for t in range(T):
            start_step_time = time.time()

            action = agent.act(state)
            if len(replay_buffer) >= cfg['begin_learning_at_step']:
                action = noisy_action(action)
            else:
                action = np.random.choice(action.shape[1],size=len(action))

            state2, reward, done, info = env.step(action.astype('int').ravel())

            replay_buffer.append({
                'state':state,
                'action':onehot(action),
                'reward':reward,
                'done':done,
                'state2':state2
            })
            state = state2 

            if len(replay_buffer) >= cfg['begin_learning_at_step'] and t % cfg['update_freq'] == 0:
                for i in range(cfg['n_update_steps']):
                    if cfg['buffer_type'] == 'prioritized':
                        beta = t*1/T
                        td_error = agent.update(learning_rate=cfg['learning_rate'],**replay_buffer.sample(cfg['batchsize'],beta=beta))
                        replay_buffer.update_priorities(td_error)
                    else:
                        agent.update(learning_rate=cfg['learning_rate'],**replay_buffer.sample(cfg['batchsize']))

            # Bookkeeping
            log['reward_history'].append(reward)
            log['action_history'].append(action)
            if t % cfg['n_steps_per_eval'] == 0 and t > 0:
                log['test_ep_returns'].append(test_agent(test_env,agent))
                log['test_ep_steps'].append(t)
                pb.add(1,[('avg_action', action.mean()),('test_ep_returns', log['test_ep_returns'][-1])])
            else:
                pb.add(1,[('avg_action', action.mean())])
            end_time = time.time()
            log['step_duration_sec'].append(end_time-start_step_time)
            log['duration_cumulative'].append(end_time-start_time)

    if cfg['buffer_type'] == 'prioritized':
        log['priorities'] = replay_buffer

    with h5py.File(os.path.join(savedir,'log.h5'), 'w') as f:
        for k in log:
            f[k] = log[k]

if __name__=='__main__':
    run()
