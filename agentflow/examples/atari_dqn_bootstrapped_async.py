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
from agentflow.utils import LogsTFSummary


@click.option('--env_id', default='PongDeterministic-v4', type=str)
@click.option('--num_steps', default=30000, type=int)
@click.option('--num_frames_max', default=None, type=int)
@click.option('--n_runners', default=1, type=int)
@click.option('--runner_update_freq', default=100, type=int)
@click.option('--n_gpus', default=0, type=int)
@click.option('--n_envs', default=1, type=int)
@click.option('--n_prev_frames', default=4, type=int)
@click.option('--ema_decay', default=0.95, type=float)
@click.option('--noise_type', default='eps_greedy', type=click.Choice(['eps_greedy','gumbel_softmax']))
@click.option('--noise_scale_anneal_steps', default=int(1e6), type=int)
@click.option('--noise_scale_init', default=1.0, type=float)
@click.option('--noise_scale_final', default=0.01, type=float)
@click.option('--noise_temperature', default=1.0, type=float)
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
@click.option('--prioritized_replay_beta_iters', default=None, type=int)
@click.option('--prioritized_replay_eps', default=1e-6, type=float)
@click.option('--prioritized_replay_simple', default=True, type=bool)
@click.option('--prioritized_replay_default_reward_priority', default=5, type=float)
@click.option('--prioritized_replay_default_done_priority', default=5, type=float)
@click.option('--begin_learning_at_step', default=200)
@click.option('--learning_rate', default=1e-4)
@click.option('--learning_rate_final', default=0.0, type=float)
@click.option('--learning_rate_decay', default=0.99995)
@click.option('--gamma', default=0.99)
@click.option('--weight_decay', default=1e-4)
@click.option('--entropy_loss_weight', default=1e-5, type=float)
@click.option('--entropy_loss_weight_decay', default=0.99995)
@click.option('--n_update_steps', default=4, type=int)
@click.option('--update_freq', default=1, type=int)
@click.option('--n_steps_per_eval', default=1000, type=int)
@click.option('--batchsize', default=64)
@click.option('--log_flush_freq', default=1000, type=int)
@click.option('--savedir', default='results')
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
    test_env = dqn_atari_paper_env(cfg['env_id'], n_envs=1, n_prev_frames=cfg['n_prev_frames'])
    test_env = TestAgentEnv(test_env)

    # state and action shapes
    state = env.reset()['state']
    state_shape = state.shape
    action_shape = env.n_actions()
    print('STATE SHAPE: ', state_shape)
    print('ACTION SHAPE: ', action_shape)

    def q_fn(state, training=False, **kwargs):
        state = state/255 - 0.5
        state, _ = normalize_ema(state, training)
        h = dqn_atari_paper_net(state, cfg['network_scale'])
        output = tf.layers.dense(h,action_shape*cfg['bootstrap_num_heads'])
        return tf.reshape(output,[-1,action_shape,cfg['bootstrap_num_heads']])

    def build_agent():
        return BootstrappedDQN(
            state_shape=state_shape[1:],
            num_actions=action_shape,
            q_fn=q_fn,
            double_q=cfg['double_q'],
            num_heads=cfg['bootstrap_num_heads'],
            random_prior=cfg['bootstrap_random_prior'],
            prior_scale=cfg['bootstrap_prior_scale'],
        )

    @ray.remote(num_cpus=1)
    class Runner(object):

        import tensorflow as tf
        import ray.experimental.tf_utils

        def __init__(self, env, log):
            self.log = log
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


        def bootstrap_mask(self):
            bootstrap_mask_probs = (1-cfg['bootstrap_mask_prob'],cfg['bootstrap_mask_prob'])
            bootstrap_mask_shape = (len(state),cfg['bootstrap_num_heads'])
            return np.random.choice(2, size=bootstrap_mask_shape, p=bootstrap_mask_probs)

        def step(self, t):
            while True:
                # do-while to initially fill the buffer
                noise_scale = self.noise_scale_schedule(t)
                action_probs = self.agent.act_probs(self.next['state'], self.next['mask'], self.sess)
                if cfg['noise_type'] == 'eps_greedy':
                    action = eps_greedy_noise(action_probs, eps=noise_scale)
                elif cfg['noise_type'] == 'gumbel_softmax':
                    action = gumbel_softmax_noise(action_probs, temperature=cfg['noise_temperature'])
                else:
                    raise NotImplementedError("unknown noise type %s" % cfg['noise_type'])

                self.prev = self.next
                self.next = env.step(action.astype('int').ravel())

                data = {
                    'state':self.prev['state'],
                    'action':onehot(action, depth=action_shape),
                    'reward':self.next['reward'],
                    'done':self.next['done'],
                    'state2':self.next['state'],
                    'mask':self.bootstrap_mask(),
                }

                self.n_step_return_buffer.append(data)
                if self.n_step_return_buffer.full():
                    # buffer is initialized (i.e. it's full)
                    break

            self.log.append_dict.remote({
                'noise_scale': noise_scale,
                'train_ep_return': self.next['prev_episode_return'],
                'train_ep_length': self.next['prev_episode_length'],
                'prev_episode_return': self.next['prev_episode_return'], # for backwards compatibility
                'prev_episode_length': self.next['prev_episode_length'], # for backwards compatibility
            })

            delayed_data, _ = self.n_step_return_buffer.latest_data() 
            return delayed_data 

        def set_weights(self, weights):
            self.variables.set_weights(weights)
            self._set_weights_counter += 1
            self.log.append.remote('set_weights/' + self._name, self._set_weights_counter)

    @ray.remote(num_cpus=1)
    class TestRunner(object):

        import tensorflow as tf

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
            test_output = self.env.test(self.agent, self.sess, addl_outputs=['Q_policy_eval'])
            self.log.append_dict.remote({
                'test_ep_returns': test_output['return'],
                'test_ep_length': test_output['length'],
                'test_ep_t': t,
                'test_ep_steps': frame_counter,
                'test_Q_policy_eval': test_output['Q_policy_eval'].mean(axis=1)
            })

        def set_weights(self, weights):
            self.variables.set_weights(weights)
            self._set_weights_counter += 1
            self.log.append.remote('set_weights/' + self._name, self._set_weights_counter)


    @ray.remote(num_cpus=1)
    class ReplayBuffer(object):

        def __init__(self, log):

            self.log = log

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
                )

                self.beta_schedule = LinearAnnealingSchedule(
                    initial_value = cfg['prioritized_replay_beta0'],
                    final_value = 1.0,
                    annealing_steps = cfg['prioritized_replay_beta_iters'] or cfg['num_steps'],
                    begin_at_step = cfg['begin_learning_at_step'],
                )
                self.t = 0

            else:
                # Normal Buffer
                replay_buffer = BufferMap(cfg['buffer_size'] // cfg['n_envs'])

            self.replay_buffer = replay_buffer

        def append(self, data):
            self.replay_buffer.append(data)
            self.log.append_dict.remote({
                'replay_buffer_size': len(self.replay_buffer),
                'replay_buffer_frames': len(self.replay_buffer)*cfg['n_envs'],
                })

        def sample(self):
            if cfg['buffer_type'] == 'prioritized':
                beta = self.beta_schedule(self.t)
                sample = self.replay_buffer.sample(cfg['batchsize'],beta=beta)
                self.log.append.remote('beta',beta)
                self.log.append.remote('max_importance_weight',sample['importance_weight'].max())
            else:
                sample = self.replay_buffer.sample(cfg['batchsize'])
            self.t += 1
            return sample

        def update_priorities(self, priorities):
            self.replay_buffer.update_priorities(priorities)

    @ray.remote(num_cpus=1, num_gpus=cfg['n_gpus'])
    class ParameterServer(object):

        import tensorflow as tf
        import ray.experimental.tf_utils

        def __init__(self, log):

            self.log = log

            # build agent
            self.agent = build_agent()
            self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
            self.sess.run(tf.global_variables_initializer())
            self.variables = ray.experimental.tf_utils.TensorFlowVariables(
                    self.agent.outputs['loss'], self.sess)

            # Annealed parameters
            self.learning_rate_schedule = ExponentialDecaySchedule(
                initial_value = cfg['learning_rate'],
                final_value = cfg['learning_rate_final'],
                decay_rate = cfg['learning_rate_decay'],
                begin_at_step = cfg['begin_learning_at_step']
            )

            self.entropy_loss_weight_schedule = ExponentialDecaySchedule(
                initial_value = cfg['entropy_loss_weight'],
                final_value = 0.0,
                decay_rate = cfg['learning_rate_decay'],
                begin_at_step = cfg['begin_learning_at_step']
            )

            self.t = 0

        def update(self, sample):
            learning_rate = self.learning_rate_schedule(self.t)
            entropy_loss_weight = self.entropy_loss_weight_schedule(self.t)
            self.log.append.remote('learning_rate', learning_rate)
            self.log.append.remote('entropy_loss_weight', entropy_loss_weight)
            self.t += 1
            update_outputs = self.agent.update(
                session = self.sess,
                learning_rate=learning_rate,
                ema_decay=cfg['ema_decay'],
                gamma=cfg['gamma'],
                weight_decay=cfg['weight_decay'],
                entropy_loss_weight=entropy_loss_weight,
                outputs=['td_error','Q_policy_eval', 'loss'],
                **sample)

            update_outputs['learning_rate'] = learning_rate
            update_outputs['entropy_loss_weight'] = entropy_loss_weight
            self.log.append_dict.remote(update_outputs)

            if cfg['buffer_type'] == 'prioritized' and not cfg['prioritized_replay_simple']:
                return update_outputs['td_error']
            else:
                return

        def get_weights(self):
            return self.variables.get_weights()

    class Task(object):

        def run(self, t=None):
            raise NotImplementedError("implement me")

    class RunnerTask(Task):

        def __init__(self, runner, replay_buffer):
            self.runner = runner
            self.replay_buffer = replay_buffer

        def run(self, t):
            data = self.runner.step.remote(t)
            return self.replay_buffer.append.remote(data)

    class UpdateAgentTask(Task):

        def __init__(self, parameter_server, replay_buffer):
            self.parameter_server = parameter_server
            self.replay_buffer = replay_buffer

        def run(self, t):
            sample = self.replay_buffer.sample.remote()
            rval = self.parameter_server.update.remote(sample)
            if cfg['buffer_type'] == 'prioritized' and not cfg['prioritized_replay_simple']:
                td_error = rval
                return self.replay_buffer.update_priorities.remote(td_error)
            else:
                return rval

    class UpdateRunnerWeightsTask(Task):

        def __init__(self, parameter_server, runners):
            self.parameter_server = parameter_server
            self.runners = runners

        def run(self, t=None):
            weights = self.parameter_server.get_weights.remote()
            return [runner.set_weights.remote(weights) for runner in runners]

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
    update_runner_weights_task = UpdateRunnerWeightsTask(parameter_server, runners + [test_runner])

    # initialize runners
    print("INITIALIZE RUNNERS")
    ray.get(update_runner_weights_task.run())

    # setup tasks 
    print("SETUP OPS")
    ops = {task.run(0): task for task in runner_tasks}

    print("BEGIN!!!")
    start_time = time.time()
    t = 0 #num update steps 
    T = cfg['num_steps']
    frame_counter = 0
    pb = tf.keras.utils.Progbar(T, stateful_metrics=['frame','update'])
    while t < T and (cfg['num_frames_max'] is None or frame_counter < cfg['num_frames_max']):
        start_step_time = time.time()

        # add update op
        if t == 0 and frame_counter >= cfg['begin_learning_at_step']:
            print("ADD UDPDATE")
            ops[update_agent_task.run(t)] = update_agent_task

        # init and update weights periodically
        if t % cfg['runner_update_freq'] == 0 and frame_counter > cfg['begin_learning_at_step']:
            ray.get(update_runner_weights_task.run())

        # evaluate
        if t % cfg['n_steps_per_eval'] == 0 and t > 0:
            ray.get(test_runner.test.remote(t, frame_counter))

        ready_op_list, _ = ray.wait(list(ops))
        for op_id in ready_op_list:
            task = ops.pop(op_id)
            if task in runner_tasks:
                frame_counter += cfg['n_envs']

            elif task == update_agent_task:
                t += 1
                pb.add(1)
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

        if t % cfg['log_flush_freq'] == 0 and t > 0:
            log.flush.remote(step=t)

        pb.add(0,[('frame', frame_counter), ('update', t)])


    print("WRITING RESULTS TO: %s" % os.path.join(savedir,'log.h5'))
    log.write.remote(os.path.join(savedir,'log.h5'))
    print("FINISHED")

if __name__=='__main__':
    click.command()(run)()
