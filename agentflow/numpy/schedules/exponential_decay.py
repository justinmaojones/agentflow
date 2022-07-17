from dataclasses import dataclass
import numpy as np

from agentflow.numpy.schedules.schedule import Schedule

@dataclass
class ExponentialDecaySchedule(Schedule):
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
    initial_value: float,
    final_value: float,
    defay_rate: float
    annealing_steps: int,
    begin_at_step: int = 0


    def __call__(self, step: int) -> float:
        step = 1.0 * np.maximum(0, step - self.begin_at_step)
        return self.initial_value + (self.final_value - self.initial_value) * (1 - self.decay_rate ** step)
