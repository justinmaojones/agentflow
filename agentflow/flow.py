from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Union

from agentflow.source import Source

@dataclass
class Flow(ABC):
    """
    A node in the flow connected to another node
    """
    source: Union[Source, Flow] 
