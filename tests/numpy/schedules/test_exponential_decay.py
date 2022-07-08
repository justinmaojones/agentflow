import numpy as np
import unittest

from agentflow.numpy.schedules import ExponentialDecaySchedule


class TestExponentialDecaySchedule(unittest.TestCase):

    def test_schedule(self):
        schedule = ExponentialDecaySchedule(
            initial_value = 1.1, 
            final_value = 0.1, 
            decay_rate = 0.5,
            begin_at_step = 10,
        )
        self.assertAlmostEqual(schedule(step=0), 1.1, places=4)
        self.assertAlmostEqual(schedule(step=10), 1.1, places=4)
        self.assertAlmostEqual(schedule(step=11), 0.6, places=4)
        self.assertAlmostEqual(schedule(step=12), 0.35, places=4)
        self.assertAlmostEqual(schedule(step=100), 0.1, places=4)

if __name__ == '__main__':
    unittest.main()
