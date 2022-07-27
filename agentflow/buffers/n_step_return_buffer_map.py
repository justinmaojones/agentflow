import numpy as np
from collections import deque
from typing import Union

from agentflow.buffers.buffer_map import BufferMap
from agentflow.buffers.flow import BufferFlow
from agentflow.buffers.source import BufferSource


class _NStepReturnDelayedPublisher(object):
    def __init__(
        self,
        n_steps=1,
        gamma=0.99,
        reward_key="reward",
        done_indicator_key="done",
        delayed_keys=["state2"],
    ):

        self._n_steps = n_steps
        self._gamma = gamma
        self._reward_key = reward_key
        self._done_indicator_key = done_indicator_key
        self._delayed_keys = delayed_keys
        self._delayed_buffer = BufferMap(n_steps)
        self._kwargs_queue = deque()
        self._returns = None
        self._dones = None
        self._discounts = gamma ** np.arange(n_steps)[:, None]

        self._delayed_data = None
        self._delayed_kwargs = None

    def compute_returns_and_dones(self):
        rewards = self._delayed_buffer._buffers[self._reward_key].tail(self._n_steps)
        dones = self._delayed_buffer._buffers[self._done_indicator_key].tail(
            self._n_steps
        )
        dones_shift_right_one = np.roll(dones, 1, axis=0)
        dones_shift_right_one[0] = 0
        mask = 1 - np.maximum.accumulate(dones_shift_right_one, axis=0)
        self.mask = mask
        returns = np.sum(rewards * self._discounts * mask, axis=0)
        return_dones = dones.max(axis=0)
        return returns, return_dones

    def full(self):
        return len(self._delayed_buffer) == self._n_steps

    def append(self, data, **kwargs):
        self._delayed_buffer.append(data)
        self._kwargs_queue.append(kwargs)
        if self.full():
            returns, dones = self.compute_returns_and_dones()
            data_to_publish = {
                self._reward_key: returns,
                self._done_indicator_key: dones,
            }
            for k in data:
                if k not in data_to_publish and k not in self._delayed_keys:
                    data_to_publish[k] = self._delayed_buffer._buffers[k].get(0)
            for k in self._delayed_keys:
                data_to_publish[k] = data[k]
            delayed_kwargs = self._kwargs_queue.popleft()

            self._delayed_data = data_to_publish
            self._delayed_kwargs = delayed_kwargs

    def latest_data(self):
        if self.full():
            return self._delayed_data, self._delayed_kwargs
        else:
            raise Exception("buffer must be full to publish latest data")

    def __len__(self):
        return len(self._delayed_buffer)


class NStepReturnBuffer(BufferFlow):
    def __init__(
        self,
        source: Union[BufferFlow, BufferSource],
        n_steps=1,
        gamma=0.99,
        reward_key="reward",
        done_indicator_key="done",
        delayed_keys=["state2"],
    ):

        self.source = source
        self._n_step_return_buffer = _NStepReturnDelayedPublisher(
            n_steps=n_steps,
            gamma=gamma,
            reward_key=reward_key,
            done_indicator_key=done_indicator_key,
            delayed_keys=delayed_keys,
        )

    def append(self, data, **kwargs):
        self._n_step_return_buffer.append(data, **kwargs)
        if self._n_step_return_buffer.full():
            delayed_data, delayed_kwargs = self._n_step_return_buffer.latest_data()
            self.source.append(delayed_data, **delayed_kwargs)

    def append_sequence(self, data):
        raise NotImplementedError(
            "NStepReturnBuffer.append_sequence is not currently supported"
        )

    def update_priorities(self, priority):
        return self.source.update_priorities(priority)

    def priorities(self):
        return self.source.priorities()
