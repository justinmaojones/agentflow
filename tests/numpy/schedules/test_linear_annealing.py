import numpy as np
import unittest

from agentflow.numpy.schedules import LinearAnnealingSchedule


class TestLinearAnnealingSchedule(unittest.TestCase):
    def test_schedule(self):
        schedule = LinearAnnealingSchedule(
            initial_value=0.0,
            final_value=1.0,
            annealing_steps=100,
            begin_at_step=10,
        )
        self.assertEqual(schedule(step=0), 0.0)
        self.assertEqual(schedule(step=10), 0.0)
        self.assertEqual(schedule(step=11), 0.01)
        self.assertEqual(schedule(step=61), 0.51)
        self.assertEqual(schedule(step=110), 1.0)
        self.assertEqual(schedule(step=200), 1.0)


if __name__ == "__main__":
    unittest.main()
