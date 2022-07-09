import numpy as np
import unittest

from agentflow.logging.logs_tf_summary import LogsTFSummary
from agentflow.logging.logs_tf_summary import LogsTFSummaryFlow

class TestLogsTFSummaryFlow(unittest.TestCase):

    def test_creation(self):
        log = LogsTFSummary("some_path")
        log_flow = log.with_prefix("prefix")
        log_flow.set_step(0)
        log_flow.append("key", 1)
        log_flow.append_seq("key2", [2,3])
        log_flow.append_dict({"key2": 4, "key3": 5})

        assert log_flow["key"] == [1]
        assert log["prefix/key"] == [1]
        assert log_flow["key2"] == [2,3,4]
        assert log["prefix/key2"] == [2,3,4]
        assert log_flow["key3"] == [5]
        assert log["prefix/key3"] == [5]
        

if __name__ == '__main__':
    unittest.main()

