from __future__ import annotations

from abc import ABC
from dataclasses import dataclass


@dataclass
class Source(ABC):
    """
    Abstract class for source nodes in the flow graph
    """

    ...
