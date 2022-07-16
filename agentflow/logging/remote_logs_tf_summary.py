import ray

from agentflow.logging.logs_tf_summary import LogsTFSummary
from agentflow.logging.scoped_logs_tf_summary import ScopedLogsTFSummary

class RemoteLogsTFSummary(LogsTFSummary):
    
    def __init__(self, savedir: str, **kwargs): 
        self.log = ray.remote(num_cpus=1)(LogsTFSummary).remote(savedir, **kwargs)

    def __getitem__(self, key):
        return self.log.__getitem__.remote(key)

    def _append(self, key, val):
        self.log._append.remote(key, val)

    def append(self, key, val):
        self.log.append.remote(key, val)

    def set_step(self, t):
        return self.log.set_step.remote(t)

    def stack(self, key=None):
        return self.log.stack.remote(key)

    def flush(self):
        self.log.flush.remote()

class RemoteScopedLogsTFSummary(ScopedLogsTFSummary):

    def __init__(self, log: RemoteLogsTFSummary, scope: str = None):
        self.log = log 
        self._scope_val = scope

    def scope(self, scope: str):
        return RemoteScopedLogsTFSummary(self, scope)

    def set_step(self, t):
        return self.log.set_step(t)

def remote_scoped_log_tf_summary(savedir: str, **kwargs):
    return RemoteScopedLogsTFSummary(
        log=RemoteLogsTFSummary(savedir, **kwargs),
        scope=None
    )
