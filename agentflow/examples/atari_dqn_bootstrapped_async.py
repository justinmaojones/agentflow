import click
import gc
import h5py
import numpy as np
import os
import ray
import tensorflow as tf
import time
import yaml 

from agentflow.env import VecGymEnv
from agentflow.agents import BootstrappedDQN
from agentflow.agents.utils import test_agent
from agentflow.buffers import BufferMap
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
from agentflow.tensorflow.ops import normalize_ema
from agentflow.transform import ImgDecoder
from agentflow.transform import ImgEncoder
from agentflow.utils import IdleTimer 
from agentflow.utils import ScopedIdleTimer
from agentflow.utils import LogsTFSummary

def timed(func):
    def wrapper(self, *args, **kwargs):
        key = func.__name__
        with self.scoped_timer.time(key):
            output = func(self, *args, **kwargs)
        timer_summary = self.scoped_timer.summary()
        self.log.append_dict.remote({k: timer_summary[k] for k in timer_summary})
        return output
    return wrapper

@click.option('--env_id', default='PongNoFrameskip-v4', type=str)
@click.option('--num_steps', default=30000, type=int)
@click.option('--num_frames_max', default=None, type=int)
@click.option('--n_runners', default=1, type=int)
@click.option('--runner_update_freq', default=100, type=int)
@click.option('--n_gpus', default=0, type=int)
@click.option('--n_envs', default=1, type=int)
@click.option('--n_prev_frames', default=4, type=int)
@click.option('--ema_decay', default=0.95, type=float)
@click.option('--target_network_copy_freq', default=None, type=int)
@click.option('--noise_type', default='eps_greedy', type=click.Choice(['eps_greedy','gumbel_softmax']))
@click.option('--noise_scale_anneal_steps', default=int(1e6), type=int)
@click.option('--noise_scale_init', default=1.0, type=float)
@click.option('--noise_scale_final', default=0.01, type=float)
@click.option('--noise_scale_test', default=0.01, type=float)
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
@click.option('--buffer_type', default='normal', type=click.Choice(['normal','prioritized','delayed','delayed_prioritized']))
@click.option('--buffer_size', default=30000, type=int)
@click.option('--n_step_return', default=8, type=int)
@click.option('--prioritized_replay_alpha', default=0.6, type=float)
@click.option('--prioritized_replay_beta0', default=0.4, type=float)
@click.option('--prioritized_replay_beta_final', default=0.4, type=float)
@click.option('--prioritized_replay_beta_iters', default=None, type=int)
@click.option('--prioritized_replay_eps', default=1e-6, type=float)
@click.option('--prioritized_replay_simple', default=True, type=bool)
@click.option('--prioritized_replay_default_reward_priority', default=5, type=float)
@click.option('--prioritized_replay_default_done_priority', default=5, type=float)
@click.option('--begin_at_step', default=0, type=int)
@click.option('--begin_learning_at_step', default=200, type=int)
@click.option('--learning_rate', default=1e-4)
@click.option('--learning_rate_final', default=0.0, type=float)
@click.option('--learning_rate_decay', default=0.99995)
@click.option('--td_loss', default='square', type=click.Choice(['square','huber']))
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
@click.option('--gc_freq', default=1000, type=int)
@click.option('--savedir', default='results')
@click.option('--restore_from_ckpt', default=None, type=str)
@click.option('--savemodel',default=False, type=bool)
@click.option('--seed',default=None, type=int)
@click.option('--ray_port',default=None, type=int)
def run(**cfg):
    ray_init_kwargs = {}
    if cfg['ray_port'] is not None:
        ray_init_kwargs['dashboard_port'] = cfg['ray_port']
    ray.init(ignore_reinit_error=True, **ray_init_kwargs)

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
    env = dqn_atari_paper_env(cfg['env_id'], n_envs=cfg['n_envs'], n_prev_frames=cfg['n_prev_frames'])
    env = PrevEpisodeReturnsEnv(env)
    env = PrevEpisodeLengthsEnv(env)
    env = RandomOneHotMaskEnv(env, dim=cfg['bootstrap_num_heads'])
    test_env = dqn_atari_paper_env(cfg['env_id'], n_envs=8, n_prev_frames=cfg['n_prev_frames'])
    test_env = TestAgentEnv(test_env)

    @ray.remote
    def encode(data):
        data = ImgEncoder('state', 2000)(data) 
        data = ImgEncoder('state2', 2000)(data) 
        return data

    @ray.remote
    def decode(sample):
        sample = ImgDecoder('state')(sample) 
        sample = ImgDecoder('state2')(sample) 
        return sample

    # state and action shapes
    state = env.reset()['state']
    state_shape = state.shape
    action_shape = env.n_actions()
    print('STATE SHAPE: ', state_shape)
    print('ACTION SHAPE: ', action_shape)

    if not cfg['dueling']:
        def q_fn(state, training=False, **kwargs):
            state = state/255 - 0.5
            if cfg['normalize_inputs']:
                state, _ = normalize_ema(state, training)
            h = dqn_atari_paper_net(state, cfg['network_scale'])
            output = tf.layers.dense(h,action_shape*cfg['bootstrap_num_heads'])
            return tf.reshape(output,[-1,action_shape,cfg['bootstrap_num_heads']])
    else:
        def q_fn(state, training=False, **kwargs):
            state = state/255 - 0.5
            if cfg['normalize_inputs']:
                state, _ = normalize_ema(state, training)
            h_val, h_adv = dqn_atari_paper_net_dueling(state, cfg['network_scale'])
            val_flat = tf.layers.dense(h_val, cfg['bootstrap_num_heads'])
            val = val_flat[:,None,:]
            adv_flat = tf.layers.dense(h_adv, action_shape*cfg['bootstrap_num_heads'])
            adv = tf.reshape(adv_flat, [-1,action_shape,cfg['bootstrap_num_heads']])
            adv_avg = tf.reduce_mean(adv, axis=-2, keepdims=True)
            adv_normed = adv - adv_avg
            return val + adv_normed 

    def build_agent():
        return BootstrappedDQN(
            state_shape=state_shape[1:],
            num_actions=action_shape,
            q_fn=q_fn,
            double_q=cfg['double_q'],
            num_heads=cfg['bootstrap_num_heads'],
            random_prior=cfg['bootstrap_random_prior'],
            prior_scale=cfg['bootstrap_prior_scale'],
            grad_clip_norm=cfg['grad_clip_norm'],
            td_loss=cfg['td_loss'],
        )

    @ray.remote(num_cpus=1)
    class Runner(object):

        import tensorflow as tf
        import ray.experimental.tf_utils

        def __init__(self, env, log):
            self.log = log
            self.scoped_timer = ScopedIdleTimer("ScopedIdleTimer/Runner", start_on_create=False)
            self.env = env
            self.next = self.env.reset()

            # build agent
            self.agent = build_agent()

            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
            self.variables = ray.experimental.tf_utils.TensorFlowVariables(
                    self.agent.outputs['loss'], self.sess)

            # Delays publishing of records to the underlying replay buffer for n steps
            # then publishes the discounted n-step return
            self.n_step_return_buffer = NStepReturnBuffer(
                n_steps=cfg['n_step_return'],
                gamma=cfg['gamma'],
            )

            self.noise_scale_schedule = LinearAnnealingSchedule(
                initial_value = cfg['noise_scale_init'],
                final_value = cfg['noise_scale_final'],
                annealing_steps = cfg['noise_scale_anneal_steps'],
                begin_at_step = 0, 
            )

            self._set_weights_counter = 0
            self._name = 'Runner'

            self.noise_scale = None
            self.update_noise_scale(t=cfg['begin_at_step'])


        def bootstrap_mask(self):
            bootstrap_mask_probs = (1-cfg['bootstrap_mask_prob'],cfg['bootstrap_mask_prob'])
            bootstrap_mask_shape = (len(state),cfg['bootstrap_num_heads'])
            return np.random.choice(2, size=bootstrap_mask_shape, p=bootstrap_mask_probs)

        def update_noise_scale(self, done=None, t=0):
            self.noise_scale = self.noise_scale_schedule(t)
            if False:
                noise_scale = self.noise_scale_schedule(t)
                eps = np.random.rand(cfg['n_envs'])*noise_scale
                if self.noise_scale is None:
                    self.noise_scale = eps
                else:
                    self.noise_scale = done*eps + (1-done)*self.noise_scale

        @timed
        def step(self, t):
            while True:
                # do-while to initially fill the buffer
                action_probs = self.agent.act_probs(self.next['state'], self.next['mask'], self.sess)
                if cfg['noise_type'] == 'eps_greedy':
                    action = eps_greedy_noise(action_probs, eps=self.noise_scale)
                elif cfg['noise_type'] == 'gumbel_softmax':
                    action = gumbel_softmax_noise(action_probs, temperature=cfg['noise_temperature'])
                else:
                    raise NotImplementedError("unknown noise type %s" % cfg['noise_type'])

                self.prev = self.next
                self.next = env.step(action.astype('int').ravel())

                self.update_noise_scale(self.next['done'], t=t)

                data = {
                    'state':self.prev['state'],
                    'action':onehot(action, depth=action_shape),
                    'reward':self.next['reward'],
                    'done':self.next['done'],
                    'state2':self.next['state'],
                    'mask':self.bootstrap_mask(),
                }

                if cfg['buffer_type'] == 'prioritized' and not cfg['prioritized_replay_simple']:
                    data['priority'] = self.agent.infer(
                        outputs=['abs_td_error'], 
                        gamma=cfg['gamma'],
                        session=self.sess,
                        **data
                    )['abs_td_error']

                self.n_step_return_buffer.append(data)
                if self.n_step_return_buffer.full():
                    # buffer is initialized
                    break

            self.log.append_dict.remote({
                'noise_scale': self.noise_scale,
                'train_ep_return': self.next['prev_episode_return'],
                'train_ep_length': self.next['prev_episode_length'],
                'prev_episode_return': self.next['prev_episode_return'], # for backwards compatibility
                'prev_episode_length': self.next['prev_episode_length'], # for backwards compatibility
            })

            delayed_data, _ = self.n_step_return_buffer.latest_data() 
            return delayed_data 

        @timed
        def set_weights(self, weights):
            self.variables.set_weights(weights)
            self._set_weights_counter += 1
            self.log.append.remote('set_weights/' + self._name, self._set_weights_counter)

    @ray.remote(num_cpus=1)
    class TestRunner(object):

        import tensorflow as tf
        import ray.experimental.tf_utils

        def __init__(self, env, log):
            self.log = log
            self.env = env
            self.agent = build_agent()
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
            self.variables = ray.experimental.tf_utils.TensorFlowVariables(
                    self.agent.outputs['loss'], self.sess)

            self._set_weights_counter = 0
            self._name = 'TestRunner'

        def test(self, t, frame_counter):
            test_output = self.env.test(self.agent, self.sess, noise_scale=cfg['noise_scale_test'])
            self.log.append_dict.remote({
                'test_ep_returns': test_output['return'],
                'test_ep_length': test_output['length'],
                'test_ep_t': t,
                'test_ep_steps': frame_counter,
            })

        def set_weights(self, weights):
            self.variables.set_weights(weights)
            self._set_weights_counter += 1
            self.log.append.remote('set_weights/' + self._name, self._set_weights_counter)


    @ray.remote(num_cpus=1)
    class ReplayBuffer(object):

        def __init__(self, log):

            self.log = log
            self.scoped_timer = ScopedIdleTimer("ScopedIdleTimer/ReplayBuffer", start_on_create=False)

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

                self.beta_schedule = LinearAnnealingSchedule(
                    initial_value = cfg['prioritized_replay_beta0'],
                    final_value = cfg['prioritized_replay_beta_final'],
                    annealing_steps = cfg['prioritized_replay_beta_iters'] or cfg['num_steps'],
                    begin_at_step = cfg['begin_learning_at_step'],
                )

            else:
                # Normal Buffer
                replay_buffer = BufferMap(cfg['buffer_size'] // cfg['n_envs'])

            self.t = cfg['begin_at_step']
            self.replay_buffer = replay_buffer

        @timed
        def append(self, data):
            self.replay_buffer.append(data)

            self.log.append_dict.remote({
                'replay_buffer_size': len(self.replay_buffer),
                'replay_buffer_frames': len(self.replay_buffer)*cfg['n_envs'],
                })

        @timed
        def sample(self):
            if cfg['buffer_type'] == 'prioritized':
                beta = self.beta_schedule(self.t)
                sample = self.replay_buffer.sample(cfg['batchsize'],beta=beta,with_indices=True)
                self.log.append_dict.remote({
                    'beta': beta,
                    'importance_weight': sample['importance_weight'],
                })
            else:
                sample = self.replay_buffer.sample(cfg['batchsize'])
            self.t += 1
            return sample

        @timed
        def update_priorities(self, priorities_and_indices):
            assert isinstance(priorities_and_indices, dict)
            assert 'priorities' in priorities_and_indices
            assert 'indices' in priorities_and_indices
            self.replay_buffer.update_priorities(**priorities_and_indices)

    @ray.remote(num_cpus=1, num_gpus=cfg['n_gpus'])
    class ParameterServer(object):

        import tensorflow as tf
        import ray.experimental.tf_utils

        def __init__(self, log):

            self.log = log
            self.scoped_timer = ScopedIdleTimer("ScopedIdleTimer/ParameterServer", start_on_create=False)

            # build agent
            self.agent = build_agent()
            self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
            self.sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()
            self.checkpoint_prefix = os.path.join(savedir,'ckpt')
            if cfg['restore_from_ckpt'] is not None:
                self.restore(cfg['restore_from_ckpt'])
            self.variables = ray.experimental.tf_utils.TensorFlowVariables(
                    self.agent.outputs['loss'], self.sess)

            # Annealed parameters
            self.learning_rate_schedule = ExponentialDecaySchedule(
                initial_value = cfg['learning_rate'],
                final_value = cfg['learning_rate_final'],
                decay_rate = cfg['learning_rate_decay'],
                begin_at_step = cfg['begin_learning_at_step']
            )

            self.t = cfg['begin_at_step']

            self.timer = IdleTimer(start_on_create=False) 

            for v in tf.trainable_variables():
                print(v)

        @timed
        def update(self, sample):
            self.timer(idle=False)
            learning_rate = self.learning_rate_schedule(self.t)
            weight_decay = cfg['weight_decay']
            gamma = cfg['gamma']

            ema_decay = cfg['ema_decay']
            if cfg['target_network_copy_freq'] is not None:
                if self.t % cfg['target_network_copy_freq'] == 0:
                    self.log.append.remote('target_network_copy', self.t)
                    ema_decay = 0.0

            if cfg['buffer_type'] == 'prioritized' and not cfg['prioritized_replay_simple']:
                indices = sample.pop('indices')

            update_outputs = self.agent.update(
                session = self.sess,
                learning_rate = learning_rate,
                ema_decay = ema_decay,
                gamma = gamma,
                weight_decay = weight_decay,
                outputs = ['abs_td_error', 'Q_policy_eval', 'loss', 'gnorm'],
                **sample)

            update_outputs['weight_decay'] = weight_decay
            update_outputs['gamma'] = gamma
            update_outputs['ema_decay'] = ema_decay
            update_outputs['learning_rate'] = learning_rate
            self.log.append_dict.remote(update_outputs)

            if cfg['log_pnorms']:
                log.append_dict.remote(agent.pnorms(self.sess))

            self.t += 1
            if cfg['buffer_type'] == 'prioritized' and not cfg['prioritized_replay_simple']:
                output = {
                    'priorities': np.abs(update_outputs['abs_td_error']),
                    'indices': indices,
                  }
            else:
                output = None
            self.timer(idle=True)
            self.log.append.remote('IdleTimer/fraction_idle/parameter_server', self.timer.fraction_idle())
            return output

        @timed
        def get_weights(self):
            return self.variables.get_weights()

        @timed
        def save(self):
            self.saver.save(self.sess, self.checkpoint_prefix)

        def restore(self, checkpoint_prefix):
            self.saver.restore(self.sess, checkpoint_prefix)

    class Task(object):

        def run(self, t=None):
            raise NotImplementedError("implement me")

    class RunnerTask(Task):

        def __init__(self, runner, replay_buffer):
            self.runner = runner
            self.replay_buffer = replay_buffer

        def run(self, t):
            data = self.runner.step.remote(t)
            data = encode.remote(data)
            return self.replay_buffer.append.remote(data)

    class UpdateAgentTask(Task):

        def __init__(self, parameter_server, replay_buffer):
            self.parameter_server = parameter_server
            self.replay_buffer = replay_buffer
            self._sample = None

        def sample(self):
            if self._sample is None:
                self._sample = decode.remote(self.replay_buffer.sample.remote())
            sample = self._sample
            self._sample = decode.remote(self.replay_buffer.sample.remote())
            return sample

        def run(self, t):
            rval = self.parameter_server.update.remote(self.sample())
            if cfg['buffer_type'] == 'prioritized' and not cfg['prioritized_replay_simple']:
                return self.replay_buffer.update_priorities.remote(rval)
            else:
                return rval

    class UpdateRunnerWeightsTask(Task):

        def __init__(self, parameter_server, runners):
            self.parameter_server = parameter_server
            self.runners = runners
            self.pending = {}
            self.weights = None

        def refresh_weights(self):
            self.weights = self.parameter_server.get_weights.remote()

        def update_pending(self):
            ready_ops, not_ready_ops = ray.wait(list(self.pending), num_returns=len(self.pending), timeout=1e-12)
            for op_id in ready_ops:
                self.pending.pop(op_id)

        def update_runner(self, runner):
            self.update_pending()
            # for each runner that is ready, update weights 
            not_ready_runners = list(self.pending.values())
            if runner in not_ready_runners:
                print("Could not update runner, since it is still pending completion of a prior scheduled task")
                return []
            else:
                task = runner.set_weights.remote(self.weights)
                self.pending[task] = runner
                return [task]

        def run(self, t):
            # get weights
            self.refresh_weights()

            # remove completed ops from pending
            self.update_pending()
            
            # for each runner that is ready, update weights 
            not_ready_runners = list(self.pending.values())
            ready_runners = [runner for runner in self.runners if runner not in not_ready_runners]
            for runner in ready_runners:
                self.pending[runner.set_weights.remote(self.weights)] = runner
            return list(self.pending)


    print("BUILD ACTORS")
    RemoteLogsTFSummary = ray.remote(LogsTFSummary)
    log = RemoteLogsTFSummary.remote(savedir)
    runners = [Runner.remote(env, log) for i in range(cfg['n_runners'])]
    test_runner = TestRunner.remote(test_env, log)
    parameter_server = ParameterServer.remote(log)
    replay_buffer = ReplayBuffer.remote(log)

    print("BUILD TASK MANAGERS")
    runner_tasks = [RunnerTask(runner, replay_buffer) for runner in runners]
    update_agent_task = UpdateAgentTask(parameter_server, replay_buffer)
    update_runner_weights_task = UpdateRunnerWeightsTask(parameter_server, runners)

    # initialize runners
    print("INITIALIZE RUNNERS")
    ray.get(update_runner_weights_task.run(0))

    # setup tasks 
    print("SETUP OPS")
    ops = {}
    for i in range(2):
        for task in runner_tasks:
            ops[task.run(i)] = task

    print("BEGIN!!!")
    start_time = time.time()
    t = cfg['begin_at_step'] #num update steps 
    T = cfg['num_steps']
    frame_counter = 0
    pb = tf.keras.utils.Progbar(T, stateful_metrics=['frame','update'])
    while t < T and (cfg['num_frames_max'] is None or frame_counter < cfg['num_frames_max']):
        start_step_time = time.time()

        # add update op
        if t == cfg['begin_at_step'] and frame_counter >= cfg['begin_learning_at_step']:
            if update_agent_task not in list(ops.values()):
                print("ADD UDPDATE")
                for i in range(16):
                    ops[update_agent_task.run(t)] = update_agent_task

        ready_op_list, _ = ray.wait(list(ops))
        for op_id in ready_op_list:
            task = ops.pop(op_id)
            if task in runner_tasks:
                frame_counter += cfg['n_envs']
                pb.add(0,[('frame', frame_counter), ('update', t)])
            elif task == update_agent_task:
                t += 1

                # update weights periodically
                if t % cfg['runner_update_freq'] == 0:
                    update_runner_weights_task.run(t)

                # evaluate
                if t % cfg['n_steps_per_eval'] == 0:
                    update_runner_weights_task.update_runner(test_runner)
                    test_runner.test.remote(t, frame_counter)

                if t % cfg['log_flush_freq'] == 0:
                    log.flush.remote(step=t)
                    if cfg['savemodel']:
                        parameter_server.save.remote()

                if t % cfg['gc_freq'] == 0:
                    gc.collect()

                pb.add(1,[('frame', frame_counter), ('update', t)])
            else:
                assert False, "unhandled worker: %s" % str(worker)

            ops[task.run(t)] = task

        end_time = time.time()
        log.append_dict.remote({
            'update': t,
            'frame': frame_counter,
            'step_duration_sec': end_time-start_step_time,
            'duration_cumulative': end_time-start_time,
        })

    print("WRITING RESULTS TO: %s" % os.path.join(savedir,'log.h5'))
    log.write.remote(os.path.join(savedir,'log.h5'))
    print("FINISHED")

if __name__=='__main__':
    click.command()(run)()
