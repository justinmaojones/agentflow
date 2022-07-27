from dataclasses import dataclass
import numpy as np
from typing import Dict

from agentflow.buffers.flow import BufferFlow
from agentflow.numpy.ops import onehot


@dataclass
class ActionToOneHotBuffer(BufferFlow):

    num_actions: int

    def append(self, data: Dict[str, np.ndarray]):
        assert "action" in data, "missing required key 'action' in data"
        data = {k: v for k, v in data.items()}
        data["action"] = onehot(data["action"], self.num_actions)
        self.source.append(data)

    def append_sequence(self, data: Dict[str, np.ndarray]):
        raise NotImplementedError
