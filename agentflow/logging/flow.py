from dataclasses import dataclass

from agentflow.logging.logs_tf_summary import LogsTFSummary
from agentflow.logging.source import WithLoggingSource
from agentflow.flow import Flow

@dataclass
class WithLoggingFlow(Flow):

    log: LogsTFSummary = None

    def set_log(self, log: LogsTFSummary):
        assert self.log is None, "cannot set log after it has already been set"
        self.log = log

        assert isinstance(self.source, WithLoggingFlow) or isinstance(self.source, WithLoggingSource), \
                "source must also be a type of WithLoggingSource or WithLoggingFlow"
        self.source.set_log(log)
