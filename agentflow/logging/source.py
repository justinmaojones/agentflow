from dataclasses import dataclass

from agentflow.logging.logs_tf_summary import LogsTFSummary
from agentflow.source import Source

@dataclass
class WithLoggingSource(Source):

    log: LogsTFSummary = None

    def set_log(self, log: LogsTFSummary):
        assert self.log is None, "cannot set log after it has already been set"
        self.log = log
