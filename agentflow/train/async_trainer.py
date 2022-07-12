from collections import Counter
from enum import Enum, auto
import numpy as np
import ray
import tensorflow as tf
import threading
import time
from typing import List, Union

from agentflow.agents import AgentFlow
from agentflow.agents import AgentSource
from agentflow.agents.utils import test_agent as test_agent_fn
from agentflow.buffers import BufferFlow
from agentflow.buffers import BufferSource
from agentflow.env import BaseEnv
from agentflow.logging import RemoteScopedLogsTFSummary
from agentflow.logging import remote_scoped_log_tf_summary
from agentflow.state import StateEnv
from agentflow.utils import ScopedIdleTimer

class Runner:

    import tensorflow as tf

    def __init__(self, :
            env: Union[BaseEnv, StateEnv], 
            agent: Union[AgentFlow, AgentSource],
            replay_buffer: Union[BufferFlow, BufferSource],
            log: RemoteScopedLogsTFSummary = None,
        ):

        self.env = env
        self.agent = agent.build()
        self.replay_buffer = replay_buffer
        self.log = log

        #self.scoped_timer = ScopedIdleTimer("ScopedIdleTimer/Runner", start_on_create=False)

        # initialize
        self.next = self.env.reset()
        self._set_weights_counter = 0
        self._frame_counter = 0

        # to ensure that buffers are thread-safe
        self._lock_buffer = threading.Lock()

    @property
    def frame_counter(self):
        return self._frame_counter

    @property
    def name(self):
        return self._name

    #@timed
    def set_weights(self, weights):
        self.agent.set_weights(weights)
        self._set_weights_counter += 1
        self.log.append('set_weights', self._set_weights_counter)

    #@timed
    def step(self):
        if self.next is None:
            self.next = self.env.reset()

        action = self.agent.act(self.next['state'])
        if isinstance(action, tf.Tensor):
            action = action.numpy()

        self.prev = self.next
        self.next = env.step(action)

        data = {
            'state': self.prev['state'],
            'action': action,
            'reward': self.next['reward'],
            'done': self.next['done'],
            'state2': self.next['state'],
        }

        with self._lock_buffer:
            self.replay_buffer.append(data)

        # num frames = num steps x num envs
        self._frame_counter += len(self.next['state'])

    def sample(self, n_samples, **kwargs):
        with self._lock_buffer:
            return self.replay_buffer.sample(n_samples, **kwargs)

    def run(self, n_steps=None):
        i = 0
        self._run = True
        while self._run:
            if n_steps is None or i < n_steps:
                self.step()
            i += 1

    def stop(self):
        self._run = False

class TestRunner:

    import tensorflow as tf

    def __init__(self, :
            env: Union[BaseEnv, StateEnv], 
            agent: Union[AgentFlow, AgentSource],
            log: RemoteScopedLogsTFSummary = None,
        ):

        self.env = env
        self.agent = agent.build()
        self.log = log

        #self.scoped_timer = ScopedIdleTimer("ScopedIdleTimer/TestRunner", start_on_create=False)
        self.next = self.env.reset()

        self._set_weights_counter = 0
        self._test_counter = 0
        
    @property
    def name(self):
        return self._name

    ##@timed
    def set_weights(self, weights):
        self.agent.set_weights(weights)
        self._set_weights_counter += 1
        self.log.append('set_weights', self._set_weights_counter)

    def test(self):
        test_output = test_agent_fn(self.env, self.agent)
        self._test_counter += 1
        self.log.append("test_counter", self._test_counter)
        return test_output



class ParameterServer:

    import tensorflow as tf

    def __init__(self, 
            agent: Union[AgentFlow, AgentSource],
            runners: List[Runner],
            batchsize: int,
            batchsize_runner: int = None,
            log: RemoteScopedLogsTFSummary
        ):

        self.agent = agent.build()
        self.runners = runners
        self.batchsize = batchsize
        self.log = log

        if batchsize_runner is None:
            self.batchsize_runner = self.batchsize
        else:
            assert self.batchsize % self.batchsize_runner == 0, \
                    f"batchsize={batchsize} must be multiple of batchsize_runner={batchsize_runner}"
            self.batchsize_runner = batchsize_runner

        self._dataset = None
        self._update_counter = 0


    def _build_dataset_pipeline(self):

        sample = self.runners[0].sample(self.batchsize)

        output_signature = {
            k: tf.TensorSpec(shape=sample[k].shape, dtype=sample[k].dtype)
            for k in sample
        }

        def sample_runner_generator(i):
            while True:
                yield self.runners[i].sample(self.batchsize_runner)

        dataset = tf.data.Dataset.range(len(self.runners))
        dataset = dataset.interleave(lambda i: 
            tf.data.Dataset.from_generator(
                sample_runner_generator,
                output_signature=output_signature,
                args=(i,)
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False,
        )

        # when batchsize >> batchsize_runner, we ensure that we get
        # samples from multiple runners in each batch
        dataset = dataset.batch(
            self.batchsize // self.batchsize_runner,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False,
        )

        # so that we aren't waiting on sample calls
        dataset = dataset.prefetch(8)
        self._dataset = dataset


    #@timed
    def get_weights(self):
        return self.agent.get_weights()

    #@timed
    def save(self):
        self.agent.save_weights(self.checkpoint_prefix)

    def restore(self, checkpoint_prefix):
        self.agent.load_weights(self.checkpoint_prefix)

    #@timed
    def update(self, n_steps: int = 1):

        if self._dataset is None:
            self._build_dataset_pipeline()

        start_time = time.time()
        t = 0 
        for sample in self._dataset:
            update_outputs = self.agent.update(**sample)
            self._update_counter += 1
            t += 1
            if t >= n_steps:
                break
        end_time = time.time()

        self.log.append('train/steps_per_sec', (t+1.) / (end_time-start_time))
        self.log.append('update_counter', self._update_counter)

        return self._update_counter

    @property
    def update_counter(self):
        return self._update_counter

class WeightUpdater:

    def __init__(self, 
            parameter_server: ParameterServer, 
            runners: List[Union[Runner, TestRunner]],
            log: RemoteScopedLogsTFSummary = None,
        ):
        self.parameter_server = parameter_server
        self.runners = runners
        self.log = log
        self._refresh_counter = 0

    def update(self)
        # get weights
        weights = self.parameter_server.get_weights.remote()

        # update runners
        ops = []
        for runner in self.runners:
            ops.append(runner.set_weights.remote(weights))

        # block until all have finished 
        ray.get(ops)

        self._refresh_counter += 1
        self.log.append("refresh_counter", self._refresh_counter)

class AsyncTrainer:

    def __init__(self, 
            env: Union[BaseEnv, StateEnv], 
            agent: Union[AgentFlow, AgentSource],
            replay_buffer: Union[BufferFlow, BufferSource],
            begin_learning_at_frame: int,
            n_updates_per_model_refresh: int
            batchsize: int,
            batchsize_runner: int = None,
            test_env: Union[BaseEnv, StateEnv] = None,
            test_agent: Union[AgentFlow, AgentSource] = None,
            runner_count: int = 1,
            runner_cpu: int = 1,
            runner_threads: int = 2,
            parameter_server_cpu: int = 1,
            parameter_server_gpu: int = 0,
            parameter_server_threads: int = 2,
            log: RemoteScopedLogsTFSummary = None,
            start_step: int = 0,
            max_frames: int = 0
        ):

        ray.init()

        self.env = env
        self.agent = agent
        self.replay_buffer = replay_buffer
        self.test_env = test_env or env
        self.test_agent = test_agent or agent

        self.runner_count = runner_count
        self.runner_cpu = runner_cpu

        self.parameter_server_cpu = parameter_server_cpu
        self.parameter_server_gpu = parameter_server_gpu

        self.batchsize = batchsize
        if batchsize_runner is None:
            self.batchsize_runner = batchsize // runner_count

        self.log = log

        self.start_step = start_step
        self.begin_learning_at_frame = begin_learning_at_frame
        self.n_updates_per_model_refresh = n_updates_per_model_refresh
        self.max_frames = max_frames

        self.set_step(start_step)

        self._frame_counter = 0
        self._update_counter = 0

        self.runners = None
        self.test_runner = None
        self.parameter_server = None

        self.build_runners()
        self.build_test_runner()
        self.build_parameter_server()
        self.build_weight_updater()

        # blocking call to initialize everything
        ray.get(self.weight_updater.update.remote())

    def build_runners(self):
        assert self.runners is None, "runners already built"

        RemoteRunner = ray.remote(
            num_cpus=self.runner_cpu
        )(Runner)

        # runners perform two general tasks: 
        # 1) interacting with environment to collect training data
        # 2) sampling from buffer and returning sample to parameter server
        # Thus we should have at least 2 threads per runner to handle IO
        # concurrently with computation for some small gains
        RemoteRunner = RemoteRunner.options(
            max_concurrency=self.runner_threads
        )

        # create runners
        self.runners = []
        for i in range(self.runner_count):
            runner = RemoteRunner.remote(
                env=self.env, 
                agent=self.agent, 
                replay_buffer=self.replay_buffer, 
                log=self.log.scope(f"runner/{i}"), 
            )
            self.runners.append(runner)

    def build_test_runner(self):
        assert self.test_runners is None, "test_runner already built"

        RemoteTestRunner = ray.remote(
            num_cpus=1
        )(TestRunner)

        self.test_runner = RemoteTestRunner.remote(
            env=self.test_env,
            agent=self.test_agent,
            log=self.log.scope("test_runner"),
        )

    def build_parameter_server(self):
        assert self.runners is not None, "need to build runners first"
        assert self.parameter_server is None, "parameter server already built"

        RemoteParameterServer = ray.remote(
            num_cpus=self.parameter_server_cpu, 
            num_gpus=self.parameter_server_gpu
        )(ParameterServer)

        # Threads should be at least 2 to overlap communication with compute.
        # However, additional threads may be warranted to support data pipelining.
        RemoteParameterServer = RemoteParameterServer.options(
            max_concurrency=self.parameter_server_threads
        )

        # create parameter server
        self.parameter_server = RemoteParameterServer.remote(
            agent=self.agent,
            runners=self.runners,
            batchsize=self.batchsize,
            batchsize_runner=self.batchsize_runner,
            log=self.log.scope("parameter_server")
        )

    def build_weight_updater(self):
        assert self.runners is not None, "need to build runners first"
        assert self.test_runner is not None, "need to build test runner first"
        assert self.parameter_server is not None, "need to build runners first"

        RemoteWeightUpdater = ray.remote(
            num_cpus=1
        )(WeightUpdater)

        self.weight_updater = RemoteWeightUpdater.remote(
            parameter_server=self.parameter_server,
            runners=self.runners + [self.test_runner],
            log=self.log.scope("weight_upater")
        )


    def remote_frame_counter(self):
        @remote
        def f():
            return sum(ray.get([r.frame_counter.remote() for r in self.runners]))
        return f.remote()

    @property
    def frame_counter(self):
        return ray.get(remote_frame_counter)

    def set_step(self, t):
        if self.log is not None:
            self.log.set_step(t)

    def start_runners(self):
        ray.get([r.run.remote() for r in self.runners()])

    def stop_runners(self):
        ray.get([r.stop.remote() for r in self.runners()])

    def learn(self, num_updates: int):

        update_counter = 0
        frame_counter = 0
        test_counter = 0
        refresh_weights_counter = 0
        num_updates = num_updates 

        progress_bar_metrics = [
            'test/ep_returns',
            'frames',
            'updates',
        ]

        pb = tf.keras.utils.Progbar(num_updates, stateful_metrics=progress_bar_metrics)
        start_time = time.time()

        self.start_runners()

        class Op(Enum):
            FRAME_COUNTER = auto()
            REFRESH_WEIGHTS = auto()
            TEST_RUN = auto()
            UPDATE_AGENT = auto()

        op_counters = Counter()

        ops = {}
        while True:

            if self.max_frames is not None:
                if frame_counter >= self.max_frames:
                    print(f"Stopping program because frame_counter={self._frame_counter} "
                          f"has exceeded num_frames_max={self.max_frames}")
                    break

            if update_counter >= num_updates:
                break

            if Op.FRAME_COUNTER not in ops.values():
                ops[self.remote_frame_counter()] = Op.FRAME_COUNTER

            if Op.TEST_RUN not in ops.values():
                if op_counter[Op.TEST_RUN] < op_counter[Op.REFRESH_WEIGHTS]:
                    # run test after every weight refresh
                    ops[self.test_runner.test.remote()] = Op.TEST_RUN

            if Op.REFRESH_WEIGHTS not in ops.values():
                if op_counter[Op.REFRESH_WEIGHTS] < op_counter[Op.UPDATE_AGENT]:
                    # refresh weights after updates
                    ops[self._weight_updater.update.remote()] = Op.REFRESH_WEIGHTS

            if Op.UPDATE_AGENT not in ops.values():
                if frame_counter >= self.begin_learning_at_frame:
                    _n_update_steps = min(num_updates-update_counter, self.n_updates_per_model_refresh)
                    ops[self.parameter_server.update.remote(_n_update_steps)] = Op.UPDATE_AGENT

            ready_op_list, _ = ray.wait(list(ops))
            for op in ready_op_list:
                op_type = ops.pop(op)
                op_counter[op_type] += 1

                assert op_type not in ops.values(), f"{op_type} op still in ops"

                if op_type == Op.FRAME_COUNTER:
                    frame_counter = ray.get(op)
                    self.log.append("frames", frame_counter)
                    pb.update(update_counter, [('frames', frame_counter)])

                elif op_type == Op.UPDATE_AGENT:
                    update_counter = ray.get(op)
                    self.set_step(update_counter)
                    pb.add(update_counter, [('updates', updates)])

                    # flush log
                    if self.log: 
                        self.log.flush()

                elif op_type == Op.REFRESH_WEIGHTS:
                    pass

                elif op_type == Op.TEST_RUN:
                    pb.update(t, [('test/ep_returns', ray.get(op))])

                else:
                    raise NotImplementedError(f"Unhandled op_type={op_type}")

        self.stop_runners()
