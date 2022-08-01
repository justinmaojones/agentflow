import numpy as np
import tensorflow as tf
import time
from typing import Union

from agentflow.agents import AgentFlow
from agentflow.agents import AgentSource
from agentflow.agents.utils import test_agent as test_agent_fn
from agentflow.buffers import BufferFlow
from agentflow.buffers import BufferSource
from agentflow.env import EnvFlow
from agentflow.env import EnvSource
from agentflow.logging import ScopedLogsTFSummary
from agentflow.logging import WithLogging
from agentflow.tensorflow.profiler import TFProfiler


class Trainer(WithLogging):
    def __init__(
        self,
        env: Union[EnvSource, EnvFlow],
        agent: Union[AgentFlow, AgentSource],
        replay_buffer: Union[BufferFlow, BufferSource],
        batchsize: int,
        test_env: Union[EnvSource, EnvFlow],
        test_agent: Union[AgentFlow, AgentSource] = None,
        log: ScopedLogsTFSummary = None,
        log_flush_freq: int = 100,
        start_step: int = 0,
        begin_learning_at_step: int = 0,
        update_freq: int = 1,
        n_update_steps: int = 1,
        n_steps_per_eval: int = 100,
        max_frames: int = None,
        profiler_start_step: int = 100,
        profiler_stop_step: int = 200,
        n_dataset_prefetches: int = 2,
        inter_op_parallelism: int = 6,
        intra_op_parallelism: int = 6,
    ):

        tf.config.threading.set_inter_op_parallelism_threads(inter_op_parallelism)
        tf.config.threading.set_intra_op_parallelism_threads(intra_op_parallelism)

        self.env = env
        self.agent = agent
        agent.build_model()
        self.replay_buffer = replay_buffer

        self.batchsize = batchsize

        self.test_env = test_env
        self.test_agent = test_agent if test_agent is not None else agent

        self.log = log
        self.log_agent = self.log.scope("agent")
        self.log_flush_freq = log_flush_freq

        if log:
            self.env.set_log(log.scope("train_env"))
            self.agent.set_log(self.log_agent)
            self.replay_buffer.set_log(log.scope("replay_buffer"))
            self.test_env.set_log(log.scope("test_env"))

        self.start_step = start_step
        self.begin_learning_at_step = begin_learning_at_step
        self.n_steps_per_eval = n_steps_per_eval
        self.update_freq = update_freq
        self.n_update_steps = n_update_steps
        self.max_frames = max_frames

        self._state = None

        self.set_step(start_step)

        self._frame_counter = 0
        self._update_counter = 0

        # ensure that profiler captures learning
        profiler_start_step = profiler_start_step + self.begin_learning_at_step
        profiler_stop_step = profiler_stop_step + self.begin_learning_at_step
        self._profiler = TFProfiler(
            profiler_start_step, profiler_stop_step, self.log.savedir
        )

        self._dataset = None
        self.n_dataset_prefetches = n_dataset_prefetches

    def _build_dataset_pipeline(self):

        sample = self.replay_buffer.sample(self.batchsize * self.n_update_steps)

        output_signature = {
            k: tf.TensorSpec(shape=sample[k].shape, dtype=sample[k].dtype)
            for k in sample
        }

        def sample_runner_generator():
            while True:
                for i in range(self.update_freq):
                    self.run_step()
                yield self.replay_buffer.sample(self.batchsize * self.n_update_steps)

        dataset = tf.data.Dataset.from_generator(
            sample_runner_generator,
            output_signature=output_signature
        )
        dataset = dataset.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x)).batch(self.batchsize)
        dataset = dataset.prefetch(self.n_dataset_prefetches)

        self._dataset = iter(dataset)


    def set_step(self, t):
        self.t = t
        if self.log is not None:
            self.log.set_step(t)

    def learn(self, num_steps: int):
        T = num_steps
        pb = tf.keras.utils.Progbar(T, stateful_metrics=["test_env/ep_returns"])
        self._start_update_time = None
        for t in range(T):

            if self.max_frames is not None:
                if self._frame_counter >= self.max_frames:
                    print(
                        f"Stopping program because frame_counter={self._frame_counter} "
                        f"has exceeded num_frames_max={self.max_frames}"
                    )
                    break

            pb_input = []

            self.train_step()

            if self.t % self.n_steps_per_eval == 0 and self.t > 0:
                test_ep_returns = self.eval_step()

                avg_test_ep_returns = np.mean(test_ep_returns)
                pb_input.append(("test_env/ep_returns", avg_test_ep_returns))
                self.log.append("test_env/ep_returns", avg_test_ep_returns)
                self.log.append("test_env/test_counter", self.t)

            pb.add(1, pb_input)

            if self.log is not None and t % self.log_flush_freq == 0 and t > 0:
                self.log.flush()

    def eval_step(self):
        return test_agent_fn(self.test_env, self.test_agent)

    def run_step(self):
        if self._state is None:
            self._state = self.env.reset()["state"]

        action = self.agent.act(self._state)
        if isinstance(action, tf.Tensor):
            action = action.numpy()

        step_output = self.env.step(action)

        data = {
            "state": self._state,
            "action": action,
            "reward": step_output["reward"],
            "done": step_output["done"],
            "state2": step_output["state"],
        }
        self.replay_buffer.append(data)

        # update state
        self._state = step_output["state"]

        # num frames = num steps x num envs
        self._frame_counter += len(self._state)
        self.log.append("trainer/frames", self._frame_counter)

        self.set_step(self.t + 1)
        self.log.append("trainer/steps", self.t)


    def update_step(self):

        if self._dataset is None:
            self._build_dataset_pipeline()

        if self._start_update_time is None:
            self._start_update_time = time.time()

        for _ in range(self.n_update_steps):
            sample = next(self._dataset)
            update_outputs = self.agent.update(**sample)
            self._update_counter += 1

        end_time = time.time()

        self.log_agent.append_dict(update_outputs)
        self.log.append("trainer/batchsize", self.batchsize)
        self.log.append(
            "trainer/updates_per_sec",
            self._update_counter / (end_time - self._start_update_time),
        )
        self.log.append(
            "trainer/training_examples_per_sec",
            (self._update_counter * self.batchsize)
            / (end_time - self._start_update_time),
        )
        self.log.append("trainer/update_counter", self._update_counter)

    def train_step(self):
        with self._profiler():
            if self.t < self.begin_learning_at_step:
                self.run_step()
            else:
                self.update_step()
