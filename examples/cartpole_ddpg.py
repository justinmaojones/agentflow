from agentflow.env import VecGymEnv
from agentflow.agents import DDPG
from agentflow.buffers import BufferMap, PrioritizedBufferMap
from agentflow.state import NPrevFramesStateEnv
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

def dense_net(x,units,layers,batchnorm=True,activation=tf.nn.relu,training=False,**kwargs):

    assert isinstance(layers,int) and layers > 0, 'layers should be a positive integer'
    assert isinstance(units,int) and units > 0, 'units should be a positive integer'

    h = x
    for l in range(layers):
        h = tf.layers.dense(h,units,**kwargs)
        h = activation(h)

        if batchnorm:
            BN = tf.layers.BatchNormalization()
            h = BN(h,training=training)
    return h

def build_net_fn(hidden_dims,hidden_layers,output_dim,batchnorm):
    def net_fn(state,training=False):
        h = state
        if batchnorm:
            BN = tf.layers.BatchNormalization()
            h = BN(h,training=training)

            h = dense_net(h,hidden_dims,hidden_layers,batchnorm=False,training=training)
        else:
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

def sigmoid(x):
    return 1./(1.+np.exp(-x))

def invert_softmax(x,axis=-1):
    return np.log(x)

def softmax(x,axis=-1):
    return np.exp(x)/np.exp(x).sum(axis=axis,keepdims=True)

def onehot(x,depth=2):
    shape = list(x.shape)+[2]
    y = np.zeros(shape)
    y[np.arange(len(x)),x] = 1.
    return y.astype('float32')

def noisy_action(action,eps=1.,clip=5e-2):
    action = np.minimum(1-clip,np.maximum(clip,action))
    logit = invert_softmax(action)
    u = np.random.rand(*action.shape)
    g = -np.log(-np.log(u))
    return (eps*g+logit).argmax(axis=-1)

@click.command()
@click.option('--num_steps', default=2e4, type=int)
@click.option('--n_envs', default=10)
@click.option('--env_id', default='CartPole-v1')
@click.option('--dqda_clipping', default=1.)
@click.option('--clip_norm', default=True, type=bool)
@click.option('--hidden_dims', default=32)
@click.option('--hidden_layers', default=2)
@click.option('--output_dim', default=2)
@click.option('--batchnorm', default=True, type=bool)
@click.option('--buffer_type', default='normal', type=click.Choice(['normal','prioritized']))
@click.option('--buffer_size', default=2**11, type=int)
@click.option('--begin_learning_at_step', default=2e3)
@click.option('--batchsize', default=100)
@click.option('--savedir', default='results')
@click.option('--seed',default=None, type=int)
def run(**kwargs):

    for k in sorted(kwargs):
        print('CONFIG: ',k,str(kwargs[k]))

    discrete = True
    T = kwargs['num_steps']
    n_envs = kwargs['n_envs']
    env_id = kwargs['env_id']
    dqda_clipping = kwargs['dqda_clipping']
    clip_norm = kwargs['clip_norm']
    hidden_dims = kwargs['hidden_dims']
    hidden_layers = kwargs['hidden_layers']
    output_dim = kwargs['output_dim']
    batchnorm = kwargs['batchnorm']
    buffer_type = kwargs['buffer_type']
    buffer_size = kwargs['buffer_size']
    begin_learning_at_step = kwargs['begin_learning_at_step']
    batchsize = kwargs['batchsize']
    savedir = kwargs['savedir']

    if kwargs['seed'] is not None:
        np.random.seed(kwargs['seed'])
        tf.set_random_seed(int(np.random.choice(int(1e6))+1))

    kwarg_hash = str(hash(str(sorted(kwargs))))
    savedir = os.path.join(savedir,'experiment' + kwarg_hash)
    os.system('mkdir -p {savedir}'.format(**locals()))

    with open(os.path.join(savedir,'config.yaml'),'w') as f:
        yaml.dump(kwargs, f)

    # Environment
    env = VecGymEnv(env_id,n_envs=n_envs)
    env = NPrevFramesStateEnv(env,n_prev_frames=4,flatten=True)

    test_env = VecGymEnv(env_id,n_envs=1)
    test_env = NPrevFramesStateEnv(test_env,n_prev_frames=4,flatten=True)

    state = env.reset()
    state_shape = state.shape
    action_shape = env.env.action_shape()

    # Agent
    policy_fn = build_policy_fn(hidden_dims,hidden_layers,output_dim,batchnorm)
    q_fn = build_q_fn(hidden_dims,hidden_layers,1,batchnorm)

    agent = DDPG(state_shape[1:],[2],policy_fn,q_fn,dqda_clipping,clip_norm,discrete=discrete)

    # Train and Test
    VARVALS = {v.name:tf.reduce_mean(tf.square(v)) for v in tf.global_variables()}

    buffer_type = 'prioritized'
    if buffer_type == 'prioritized':
        replay_buffer = PrioritizedBufferMap(buffer_size,alpha=0.5,eps=0.1)
    else:
        replay_buffer = BufferMap(buffer_size)

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

        pb = tf.keras.utils.Progbar(T)
        for t in range(T):

            start_step_time = time.time()

            action = agent.act(state)
            if len(replay_buffer) >= begin_learning_at_step:
                action = noisy_action(action)
            else:
                action = np.random.choice(2,size=len(action))

            state2, reward, done, info = env.step(action.astype('int').ravel())

            log['reward_history'].append(reward)
            log['action_history'].append(action)

            replay_buffer.append({'state':state,'action':onehot(action),'reward':reward,'done':done,'state2':state2})
            state = state2

            if len(replay_buffer) >= begin_learning_at_step:
                for i in range(40):
                    if buffer_type == 'prioritized':
                        beta = t*1./T
                        td_error = agent.update(learning_rate=1e-4,**replay_buffer.sample(batchsize,beta=beta))
                        replay_buffer.update_priorities(td_error)
                    else:
                        agent.update(learning_rate=1e-4,**replay_buffer.sample(batchsize))


            if t % 100 == 0 and t > 0:
                log['test_ep_returns'].append(test_agent(test_env,agent))
                log['test_ep_steps'].append(t)

                pb.add(1,[('avg_action', action.mean()),('test_ep_returns', log['test_ep_returns'][-1])])
            else:

                pb.add(1,[('avg_action', action.mean())])

            end_time = time.time()
            log['step_duration_sec'].append(end_time-start_step_time)
            log['duration_cumulative'].append(end_time-start_time)

    with h5py.File(os.path.join(savedir,'log.h5'), 'w') as f:
        for k in log:
            f[k] = log[k]

if __name__=='__main__':
    run()
