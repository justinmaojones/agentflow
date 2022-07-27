import numpy as np
from typing import Union

from agentflow.buffers.flow import BufferFlow
from agentflow.buffers.buffer_map import BufferMap
from agentflow.buffers.prioritized_buffer_map import PrioritizedBufferMap


class _DelayedBufferMapPublisher(BufferMap):
    def __init__(
        self, max_length=2**20, publish_indicator_key="done", add_return_loss=False
    ):
        super(_DelayedBufferMapPublisher, self).__init__(max_length)
        self._publish_indicator_key = publish_indicator_key
        self._count_since_last_publish = None
        self._published = []
        self.add_return_loss = add_return_loss

    def compute_returns(self, data, gamma=0.99):
        rewards = data["reward"]
        T = rewards.shape[-1]
        returns = []
        R = 0
        for t in reversed(range(T)):
            r = rewards[..., t : t + 1]
            R = r + gamma * R
            returns.append(R)
        return np.concatenate(returns[::-1], axis=-1)

    def publish(self, data):
        if self._count_since_last_publish is None:
            self._count_since_last_publish = np.zeros_like(
                data[self._publish_indicator_key]
            ).astype(int)
        self._count_since_last_publish += 1
        assert (
            self._count_since_last_publish.max() <= self.max_length
        ), "delay cannot exceed size of buffer"

        should_publish = data[self._publish_indicator_key]
        assert (
            should_publish.ndim == 1
        ), "expected data['%s'] to have ndim==1, but found ndim==%d" % (
            self._publish_indicator_key,
            should_publish.ndim,
        )

        output = []
        if np.sum(should_publish) > 0:
            # at least one record should be published
            idx = np.arange(len(should_publish))[should_publish == 1]
            for i in idx:
                # cannot retrieve sequence larger than buffer size
                seq_size = min(self._count_since_last_publish[i], self._n)
                published = self.tail(seq_size, batch_idx=[i])
                if self.add_return_loss:
                    published["returns"] = self.compute_returns(published)
                output.append(published)
                self._count_since_last_publish[i] = 0
        return output

    def append(self, data):
        super(_DelayedBufferMapPublisher, self).append(data)
        return self.publish(data)


class DelayedBufferMap(BufferFlow):
    def __init__(
        self,
        source: Union[BufferFlow, BufferMap],
        delayed_buffer_max_length: int = 2**20,
        publish_indicator_key: str = "done",
        add_return_loss: bool = False,
    ):
        self.source = source
        self._publish_indicator_key = publish_indicator_key
        self._delayed_buffer_max_length = delayed_buffer_max_length
        self._delayed_buffer_map = _DelayedBufferMapPublisher(
            self._delayed_buffer_max_length, publish_indicator_key, add_return_loss
        )

    def append(self, data):
        published = self._delayed_buffer_map.append(data)
        for seq in published:
            self.source.append_sequence(seq)

    def append_sequence(self, data):
        raise NotImplementedError(
            "DelayedBufferMap.append_sequence is not currently supported"
        )
