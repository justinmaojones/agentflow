from dataclasses import dataclass
import numpy as np
from typing import Dict

from agentflow.buffers.flow import BufferFlow

@dataclass
class BootstrapMaskBuffer(BufferFlow):

    depth: int
    sample_prob: float = 0.5

    def append(self, data: Dict[str, np.ndarray]):
        assert 'state' in data, "missing required key 'state' in data"
        mask_probs = (1-self.sample_prob, self.sample_prob)
        mask_shape = (len(data['state']), self.depth) 
        mask = np.random.choice(2, size=mask_shape, p=mask_probs)

        data = {k: v for k, v in data.items()}
        data['mask'] = mask

        self.source.append(data)

    def append_sequence(self, data: Dict[str, np.ndarray]):
        raise NotImplementedError 

