
class LinearAnnealingSchedule(object):

  def __init__(
      self,
      initial_value,
      final_value,
      annealing_steps,
      begin_at_step=0):
    """
    Applies a linear annealing strategy. It is computed as:
    ```python
    def decayed_learning_rate(step):
      step = min(max(0, step - begin_at_step), annealing_steps)
      return initial_value + (final_value - initial_value) * (step / annealing_steps) 
    ```

    For example:
    ```python
    >>> schedule = LinearAnnealingSchedule(initial_value = 0.0, final_value = 1.0, annealing_steps = 100)
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
    self.annealing_steps = annealing_steps
    self.begin_at_step = begin_at_step

  def __call__(self, step):
      step = float(min(max(0, step - self.begin_at_step), self.annealing_steps))
      return self.initial_value + (self.final_value - self.initial_value) * (step / self.annealing_steps)

if __name__ == '__main__':
    import unittest

    class Test(unittest.TestCase):

        def test_schedule(self):
            schedule = LinearAnnealingSchedule(
                initial_value = 0.0, 
                final_value = 1.0, 
                annealing_steps = 100,
                begin_at_step = 10,
            )
            self.assertEqual(schedule(step=0),0.0)
            self.assertEqual(schedule(step=10),0.0)
            self.assertEqual(schedule(step=11),0.01)
            self.assertEqual(schedule(step=61),0.51)
            self.assertEqual(schedule(step=110),1.0)
            self.assertEqual(schedule(step=200),1.0)

    unittest.main()

