import numpy as np

class ExponentialDecaySchedule(object):

  def __init__(
      self,
      initial_value,
      final_value,
      decay_rate,
      begin_at_step=0):
    """
    Applies an exponentially decayed annealing strategy. For example:
    ```python
    >>> schedule = ExponentialDecaySchedule(initial_value = 0.0, final_value = 1.0, annealing_steps = 100)
    >>> print(schedule(step=1))
    0.01
    ```

    Parameters
    ----------
    initial_value : int, float
        Initial schedule value
    final_value : int, float
        Final schedule value
    annealing_steps : int, float
        Number of steps to anneal over
    begin_at_step : int, float
        Schedule returns initial value until step `begin_at_step`
    """
    self.initial_value = initial_value
    self.final_value = final_value
    self.decay_rate = decay_rate
    self.begin_at_step = begin_at_step

  def __call__(self, step):
      step = 1.0 * np.maximum(0, step - self.begin_at_step)
      return self.initial_value + (self.final_value - self.initial_value) * (1 - self.decay_rate ** step)

if __name__ == '__main__':
    import unittest

    class Test(unittest.TestCase):

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

    unittest.main()

