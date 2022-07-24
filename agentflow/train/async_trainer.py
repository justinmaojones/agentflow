from collections import Counter
from enum import Enum, auto
import math
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
from agentflow.env import EnvFlow
from agentflow.env import EnvSource
from agentflow.logging import ScopedLogsTFSummary
from agentflow.tensorflow.profiler import TFProfilerIterator

class Runner:

    def __init__(self,
            env: Union[EnvSource, EnvFlow], 
            agent: Union[AgentFlow, AgentSource],
            replay_buffer: Union[BufferFlow, BufferSource],
            log: ScopedLogsTFSummary,
        ):

        self.env = env
        self.agent = agent
        agent.build_model()
        self.replay_buffer = replay_buffer
        self.log = log
        self.set_step(0)

        self.env.set_log(log.scope("train_env"))
        self.agent.set_log(log.scope("train_agent"))
        self.replay_buffer.set_log(log.scope("replay_buffer"))


        # initialize
        self.next = self.env.reset()
        self._set_weights_counter = 0
        self._frame_counter = 0
        self._step_counter = 0

        # to ensure that buffers are thread-safe
        self._lock_buffer = threading.Lock()

    def frame_counter(self):
        return self._frame_counter

    def step_counter(self):
        return self._step_counter

    def counters(self):
        return {'frame_counter': self._frame_counter, "step_counter": self._step_counter}

    def set_step(self, t, flush=False):
        if flush:
            self.log.flush()
        self.log.set_step(t)

    def set_weights(self, weights):
        self.agent.set_weights(weights)
        self._set_weights_counter += 1
        self.log.append('async/runner/set_weights', self._set_weights_counter)

    def step(self):
        if self.next is None:
            self.next = self.env.reset()

        action = self.agent.act(self.next['state'])
        if isinstance(action, tf.Tensor):
            action = action.numpy()

        self.prev = self.next
        self.next = self.env.step(action)

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
        self._step_counter += 1

    def sample(self, n_samples, **kwargs):
        """
        Returns a sample from the buffer.  Sample is converted to tf.Tensor
        in order to avoid expensive memory copy ops on parameter servers.
        """
        with self._lock_buffer:
            x = self.replay_buffer.sample(n_samples, **kwargs)
        return {k: tf.convert_to_tensor(v) for k, v in x.items()}

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

    def __init__(self,
            env: Union[EnvSource, EnvFlow], 
            agent: Union[AgentFlow, AgentSource],
            log: ScopedLogsTFSummary = None,
        ):

        self.env = env
        self.agent = agent
        agent.build_model()
        self.log = log
        self.set_step(0)

        self.env.set_log(log.scope("test_env"))
        self.agent.set_log(log.scope("test_agent"))

        self.next = self.env.reset()

        self._set_weights_counter = 0
        self._test_counter = 0

    def set_step(self, t, flush=False):
        if flush:
            self.log.flush()
        self.log.set_step(t)
        
    def set_weights(self, weights):
        self.agent.set_weights(weights)
        self._set_weights_counter += 1
        self.log.append('async/test_runner/set_weights', self._set_weights_counter)

    def test(self):
        test_output = test_agent_fn(self.env, self.agent)
        self._test_counter += 1
        self.log.append("test_env/test_counter", self._test_counter)
        self.log.append("test_env/ep_returns", test_output)
        return test_output



class ParameterServer:


    def __init__(self, 
            agent: Union[AgentFlow, AgentSource],
            runners: List[Runner],
            log: ScopedLogsTFSummary,
            batchsize: int,
            dataset_prefetch: int = 1,
            min_parallel_sample_rpc: int = 8,
            profiler_start_step: int = 100,
            profiler_stop_step: int = 200,
            inter_op_parallelism: int = 6,
            intra_op_parallelism: int = 6
        ):

        import tensorflow as tf
        tf.config.threading.set_inter_op_parallelism_threads(inter_op_parallelism)
        tf.config.threading.set_intra_op_parallelism_threads(intra_op_parallelism)

        self.agent = agent
        agent.build_model()
        self.runners = runners
        self.batchsize = batchsize
        self.dataset_prefetch = dataset_prefetch
        self.min_parallel_sample_rpc = min_parallel_sample_rpc
        self.log = log.scope("train_agent")
        self.set_step(0)

        self.agent.set_log(log)

        self._dataset = None
        self._update_counter = 0

        self._profiler = TFProfilerIterator(profiler_start_step, profiler_stop_step, self.log.savedir)

        self._time = None
        self._idle_time = 0
        self._running_time = 0


    def _build_dataset_pipeline(self):

        sample = ray.get(self.runners[0].sample.remote(self.batchsize))

        output_signature = {
            k: tf.TensorSpec(shape=sample[k].shape, dtype=sample[k].dtype)
            for k in sample
        }

        def sample_runner_generator(i):
            while True:
                yield ray.get(self.runners[i].sample.remote(self.batchsize))

        # ensure there are enough parallel fetches to support prefetch
        n_rpc = max(len(self.runners), self.dataset_prefetch, self.min_parallel_sample_rpc)
        
        # ensure n is multiple of len(self.runners)
        n_rpc = int(math.ceil(n_rpc / len(self.runners)) * len(self.runners))

        # rpc index
        dataset = tf.data.Dataset.range(n_rpc)

        # interleave results from different runners
        dataset = dataset.interleave(lambda i: 
            tf.data.Dataset.from_generator(
                sample_runner_generator,
                output_signature=output_signature,
                args=(i % len(self.runners),)
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False,
        )

        if self.dataset_prefetch > 0:
            # so that we aren't waiting on RPC 
            dataset = dataset.prefetch(self.dataset_prefetch)

        self._dataset = dataset


    def get_weights(self):
        return self.agent.get_weights()

    def set_step(self, t, flush=False):
        if flush:
            self.log.flush()
        self.log.set_step(t)

    def save(self):
        self.agent.save_weights(self.checkpoint_prefix)

    def restore(self, checkpoint_prefix):
        self.agent.load_weights(self.checkpoint_prefix)

    def log_time(self):
        frac_idle = self._idle_time / (self._idle_time + self._running_time + 1e-12)
        self.log.append('fraction_idle', frac_idle)

    def update(self, n_steps: int = 1):

        if self._dataset is None:
            self._build_dataset_pipeline()

        t = 0 
        for sample in self._profiler(self._dataset):

            if self._time is None:
                self._time = time.time()
            else:
                curr_time = time.time()
                self._idle_time += curr_time - self._time
                self._time = curr_time

            update_outputs = self.agent.update(**sample)

            curr_time = time.time()
            self._running_time += curr_time - self._time
            self._time = curr_time

            self._update_counter += 1
            t += 1
            if t >= n_steps:
                break

        self.log_time()
        self.log.append_dict(update_outputs)

        return self._update_counter

    @property
    def update_counter(self):
        return self._update_counter

class WeightUpdater:

    def __init__(self, 
            parameter_server: ParameterServer, 
            runners: List[Union[Runner, TestRunner]],
            log: ScopedLogsTFSummary = None,
        ):
        self.parameter_server = parameter_server
        self.runners = runners
        self.log = log.scope("async/weight_updater")
        self.set_step(0)

        self._refresh_counter = 0

    def set_step(self, t, flush=False):
        if flush:
            self.log.flush()
        self.log.set_step(t)

    def update(self):
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

class RunnerCountersServer:

    def __init__(self, runners: List[Runner]):
        self.runners = runners

    def counters(self):
        counters_list = ray.get([r.counters.remote() for r in self.runners])
        counters = {
            'frame_counter': sum([c['frame_counter'] for c in counters_list]),
            'step_counter_min': min([c['step_counter'] for c in counters_list]),
            'step_counter_max': max([c['step_counter'] for c in counters_list]),
        }
        return counters


class AsyncTrainer:

    def __init__(self, 
            env: Union[EnvSource, EnvFlow], 
            agent: Union[AgentFlow, AgentSource],
            replay_buffer: Union[BufferFlow, BufferSource],
            log: ScopedLogsTFSummary,
            begin_learning_at_step: int,
            n_updates_per_model_refresh: int,
            batchsize: int,
            dataset_prefetch: int = 1,
            min_parallel_sample_rpc: int = 8,
            test_env: Union[EnvSource, EnvFlow] = None,
            test_agent: Union[AgentFlow, AgentSource] = None,
            runner_count: int = 1,
            runner_cpu: int = 1,
            runner_threads: int = 2,
            parameter_server_cpu: int = 1,
            parameter_server_gpu: int = 0,
            parameter_server_threads: int = 2,
            start_step: int = 0,
            max_frames: int = None,
            profiler_start_step: int = 100,
            profiler_stop_step: int = 200,
        ):

        self.env = env
        self.agent = agent
        self.replay_buffer = replay_buffer
        self.test_env = test_env or env
        self.test_agent = test_agent or agent

        self.runner_count = runner_count
        self.runner_cpu = runner_cpu
        self.runner_threads = runner_threads

        self.parameter_server_cpu = parameter_server_cpu
        self.parameter_server_gpu = parameter_server_gpu
        self.parameter_server_threads = parameter_server_threads

        self.batchsize = batchsize
        self.dataset_prefetch = dataset_prefetch
        self.min_parallel_sample_rpc = min_parallel_sample_rpc

        self.log = log

        self.start_step = start_step
        self.begin_learning_at_step = begin_learning_at_step
        self.n_updates_per_model_refresh = n_updates_per_model_refresh
        self.max_frames = max_frames

        self.profiler_start_step = profiler_start_step
        self.profiler_stop_step = profiler_stop_step

        self._frame_counter = 0
        self._update_counter = 0

        self.runners = None
        self.test_runner = None
        self.parameter_server = None
        self.weight_updater = None
        self.runner_counters_server = None

        print("AsyncTrainer: build runners")
        self.build_runners()

        print("AsyncTrainer: build test runner")
        self.build_test_runner()

        print("AsyncTrainer: build parameter server")
        self.build_parameter_server()

        print("AsyncTrainer: build weight updater")
        self.build_weight_updater()

        print("AsyncTrainer: build frame counter server")
        self.build_runner_counters_server() 

        # wait 10 seconds for actors to startup
        time.sleep(10)

        print("AsyncTrainer: Set Step")
        ray.get(self.set_step(start_step))

        # blocking call to initialize everything
        print("AsyncTrainer: Broadcast initial weights")
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
                log=self.log.with_filename(f"runner_{i}_log.h5"), 
            )
            self.runners.append(runner)

    def build_test_runner(self):
        assert self.test_runner is None, "test_runner already built"

        RemoteTestRunner = ray.remote(
            num_cpus=1
        )(TestRunner)

        RemoteTestRunner = RemoteTestRunner.options(
            max_concurrency=self.runner_threads
        )

        self.test_runner = RemoteTestRunner.remote(
            env=self.test_env,
            agent=self.test_agent,
            log=self.log.with_filename("test_runner_log.h5"),
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
            dataset_prefetch=self.dataset_prefetch,
            min_parallel_sample_rpc=self.min_parallel_sample_rpc,
            log=self.log.with_filename("parameter_server_log.h5"),
            inter_op_parallelism=self.parameter_server_cpu,
            intra_op_parallelism=self.parameter_server_cpu,
        )

    def build_weight_updater(self):
        assert self.runners is not None, "need to build runners first"
        assert self.test_runner is not None, "need to build test runner first"
        assert self.parameter_server is not None, "need to build runners first"
        assert self.weight_updater is None, "weight updater already built"

        RemoteWeightUpdater = ray.remote(
            num_cpus=1
        )(WeightUpdater)

        self.weight_updater = RemoteWeightUpdater.remote(
            parameter_server=self.parameter_server,
            runners=self.runners + [self.test_runner],
            log=self.log.with_filename("weight_updater_log.h5")
        )

    def build_runner_counters_server(self):
        assert self.runners is not None, "need to build runners first"
        assert self.runner_counters_server is None, "frame counter server already built"

        RemoteRunnerCountersServer = ray.remote(
            num_cpus=1
        )(RunnerCountersServer)

        self.runner_counters_server = RemoteRunnerCountersServer.remote(
            runners=self.runners
        ) 

    def set_step(self, t, flush: bool = False):
        if flush:
            self.log.flush()
        self.log.set_step(t)
        rpc = [r.set_step.remote(t, flush) for r in self.runners] 
        rpc += [self.parameter_server.set_step.remote(t, flush)]
        rpc += [self.test_runner.set_step.remote(t, flush)]
        rpc += [self.weight_updater.set_step.remote(t, flush)]
        return rpc

    def start_runners(self):
        for r in self.runners:
            r.run.remote()

    def stop_runners(self):
        ray.get([r.stop.remote() for r in self.runners])

    def learn(self, num_updates: int):

        update_counter = 0
        frame_counter = 0
        step_counter = 0
        test_counter = 0
        refresh_weights_counter = 0
        num_updates = num_updates 

        progress_bar_metrics = [
            'test/ep_returns',
            'steps',
            'frames',
            'updates',
        ]

        pb = tf.keras.utils.Progbar(num_updates, stateful_metrics=progress_bar_metrics)
        start_time = time.time()
        start_update_time = None 

        print("AsyncTrainer: start runners")
        self.start_runners()

        class Op(Enum):
            RUNNER_COUNTERS = auto()
            REFRESH_WEIGHTS = auto()
            TEST_RUN = auto()
            UPDATE_AGENT = auto()

        op_counter = Counter()

        ops = {}
        while True:

            if self.max_frames is not None:
                if frame_counter >= self.max_frames:
                    print(f"Stopping program because frame_counter={self._frame_counter} "
                          f"has exceeded num_frames_max={self.max_frames}")
                    break

            if update_counter >= num_updates:
                break

            if Op.RUNNER_COUNTERS not in ops.values():
                ops[self.runner_counters_server.counters.remote()] = Op.RUNNER_COUNTERS

            if Op.TEST_RUN not in ops.values():
                if op_counter[Op.TEST_RUN] < op_counter[Op.REFRESH_WEIGHTS]:
                    # run test after every weight refresh
                    ops[self.test_runner.test.remote()] = Op.TEST_RUN

            if Op.REFRESH_WEIGHTS not in ops.values():
                if op_counter[Op.REFRESH_WEIGHTS] < op_counter[Op.UPDATE_AGENT]:
                    # refresh weights after updates
                    ops[self.weight_updater.update.remote()] = Op.REFRESH_WEIGHTS

            if Op.UPDATE_AGENT not in ops.values():
                if step_counter >= self.begin_learning_at_step:
                    if start_update_time is None:
                        start_update_time = time.time()
                    _n_update_steps = min(num_updates-update_counter, self.n_updates_per_model_refresh)
                    ops[self.parameter_server.update.remote(_n_update_steps)] = Op.UPDATE_AGENT

            ready_op_list, _ = ray.wait(list(ops))
            for op in ready_op_list:
                op_type = ops.pop(op)
                op_counter[op_type] += 1
                self.log.append(f"async/op_counter/{op_type}", op_counter[op_type])

                assert op_type not in ops.values(), f"{op_type} op still in ops"

                if op_type == Op.RUNNER_COUNTERS:
                    counters = ray.get(op)
                    frame_counter = counters['frame_counter']
                    step_counter = counters['step_counter_min']
                    self.log.append("trainer/frames", frame_counter)
                    self.log.append("trainer/steps/min", step_counter)
                    self.log.append("trainer/steps/max", counters['step_counter_max'])
                    pb.update(update_counter, [('frames', frame_counter)])
                    pb.update(update_counter, [('steps', step_counter)])

                elif op_type == Op.UPDATE_AGENT:
                    update_counter = ray.get(op)
                    # ensure all workers have up-to-date update counter
                    ray.get(self.set_step(update_counter, flush=True))
                    pb.update(update_counter, [('updates', update_counter)])

                    curr_time = time.time()
                    self.log.append('trainer/batchsize', self.batchsize)
                    self.log.append('trainer/updates_per_sec', update_counter / (curr_time-start_update_time))
                    self.log.append('trainer/training_examples_per_sec', (update_counter * self.batchsize) / (curr_time-start_update_time))
                    self.log.append('trainer/update_counter', update_counter)


                elif op_type == Op.REFRESH_WEIGHTS:
                    pass

                elif op_type == Op.TEST_RUN:
                    pb.update(update_counter, [('test/ep_returns', ray.get(op))])

                else:
                    raise NotImplementedError(f"Unhandled op_type={op_type}")

        self.stop_runners()
