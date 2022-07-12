from dataclasses import dataclass, field

from agentflow.logging.logs_tf_summary import LogsTFSummary

@dataclass
class WithLogging:

    log: LogsTFSummary = field(default=None, init=False)

    def set_log(self, log: LogsTFSummary):
        assert self.log is None, "cannot set log after it has already been set"
        self.log = log
