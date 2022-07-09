import numpy as np
import tensorflow as tf
import time
from typing import Union

from agentflow.agents import AgentFlow
from agentflow.agents import AgentSource
from agentflow.agents.utils import test_agent as test_agent_fn
from agentflow.buffers import BufferFlow
from agentflow.buffers import BufferSource
from agentflow.env import BaseEnv
from agentflow.logging import LogsTFSummary
from agentflow.state import StateEnv

class Trainer:

    def __init__(self, 
            env: Union[BaseEnv, StateEnv], 
            agent: Union[AgentFlow, AgentSource],
            replay_buffer: Union[BufferFlow, BufferSource],
            batchsize: int,
            test_env: Union[BaseEnv, StateEnv] = None,
            test_agent: Union[AgentFlow, AgentSource] = None,
            log: LogsTFSummary = None,
            log_flush_freq: int = 1,
            start_step: int = 0,
            begin_learning_at_step: int = 0,
            update_freq: int = 1,
            n_update_steps: int = 1,
            n_steps_per_eval: int = 100,
            max_frames: int = None,
        ):

        self.env = env
        self.agent = agent
        self.replay_buffer = replay_buffer

        self.batchsize = batchsize

        self.test_env = test_env if test_env is not None else env
        self.test_agent = test_agent if test_agent is not None else agent

        self.log = log
        self.log_flush_freq = log_flush_freq

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

    def set_step(self, t):
        self.t = t
        if self.log is not None:
            self.log.set_step(t)

    def learn(self, num_steps: int, progress_bar_metrics=['test/ep_returns']):
        T = num_steps
        pb = tf.keras.utils.Progbar(T, stateful_metrics=progress_bar_metrics)
        start_time = time.time()
        for t in range(T):

            if self.max_frames is not None:
                if self._frame_counter >= self.max_frames:
                    print(f"Stopping program because frame_counter={self._frame_counter} "
                          f"has exceeded num_frames_max={self.max_frames}")
                    break

            start_step_time = time.time()
            pb_input = []

            self.train_step()
            if self.t % self.n_steps_per_eval == 0 and self.t > 0:
                test_ep_returns = self.eval_step()

                avg_test_ep_returns = np.mean(test_ep_returns)
                pb_input.append(('test/ep_returns', avg_test_ep_returns))
                self.log.append('test/ep_returns', avg_test_ep_returns) 
                self.log.append('test/ep_steps', self.t)

            end_time = time.time()
            self.log.append('train/step_duration_sec', end_time-start_step_time)
            self.log.append('train/duration_cumulative', end_time-start_time)
            self.log.append('train/steps_per_sec', (t+1.) / (end_time-start_time))

            pb.add(1, pb_input)

            if self.log is not None and t % self.log_flush_freq == 0 and t > 0:
                self.log.flush()

    def eval_step(self):
        return test_agent_fn(self.test_env, self.test_agent)

    def run_step(self):
        if self._state is None:
            self._state = self.env.reset()['state']

        action = self.agent.act(self._state)
        if isinstance(action, tf.Tensor):
            action = action.numpy()

        step_output = self.env.step(action)

        data = {
            'state':self._state,
            'action':action,
            'reward':step_output['reward'],
            'done':step_output['done'],
            'state2':step_output['state'],
        }
        self.replay_buffer.append(data)

        # update state
        self._state = step_output['state']

        # num frames = num steps x num envs
        self._frame_counter += len(self._state)
        self.log.append('train/frames', self._frame_counter)

    def update_step(self):
        for i in range(self.n_update_steps):
            sample = self.replay_buffer.sample(self.batchsize)
            update_outputs = self.agent.update(**sample)
            self._update_counter += 1

        self.log.append_dict(update_outputs)
        self.log.append('train/updates', self._update_counter)

    def train_step(self):
        self.run_step()
        if self.t >= self.begin_learning_at_step:
            if self.t % self.update_freq == 0:
                self.update_step()

        self.set_step(self.t+1)
        self.log.append('train/steps', self.t)
