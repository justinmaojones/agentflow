import numpy as np
from typing import Dict, List, Union

from agentflow.logging.logs_tf_summary import LogsTFSummary


class ScopedLogsTFSummary(LogsTFSummary):
    def __init__(self, log: LogsTFSummary, scope: str = None):
        self.log = log
        self._scope_name = scope

    def _scoped_key(self, key: str):
        if self._scope_name:
            return f"{self._scope_name}/{key}"
        else:
            return key

    def __getitem__(self, key: str):
        return self.log[self._scoped_key(key)]

    def append(self, key: str, val: Union[float, int, np.ndarray]):
        self.log.append(self._scoped_key(key), val)

    def extend(self, key: str, vals: List[Union[float, int, np.ndarray]]):
        self.log.extend(self._scoped_key(key), vals)

    def append_dict(self, inp: Dict[str, Union[float, int, np.ndarray]]):
        self.log.append_dict({self._scoped_key(k): v for k, v in inp.items()})

    @property
    def savedir(self):
        return self.log.savedir

    def scope(self, scope_name: str):
        return ScopedLogsTFSummary(self, scope_name)

    def set_step(self, t: int):
        self.log.set_step(t)

    def stack(self, key: str = None):
        if key is not None:
            key = self._scoped_key(key)
        return self.log.stack(key)

    def flush(self):
        self.log.flush()

    def with_filename(self, filename: str):
        return ScopedLogsTFSummary(self.log.with_filename(filename), self._scope_name)


def scoped_log_tf_summary(savedir: str, **kwargs):
    return ScopedLogsTFSummary(log=LogsTFSummary(savedir, **kwargs), scope=None)
