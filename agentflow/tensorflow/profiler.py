import contextlib

import tensorflow as tf

class TFProfiler:
    """
    Wrapper around tensorflow profiler.
    """

    def __init__(self, start_step, stop_step, logdir):
        self._start_step = start_step
        self._stop_step = stop_step
        self._logdir = logdir

        self._t = 0

    @contextlib.contextmanager
    def __call__(self):
        if self._start_step <= self._t and self._t < self._stop_step:
            if self._t == self._start_step:
                tf.profiler.experimental.start(self._logdir)

            with tf.profiler.experimental.Trace(name="train", step_num=self._t, _r=1):
                yield
        else:
            yield

        self._t += 1

        if self._t == self._stop_step:
            tf.profiler.experimental.stop()

class TFProfilerIterator:

    def __init__(self, start_step, stop_stop, logdir):
        self._profiler = TFProfiler(start_step, stop_stop, logdir)

    def __call__(self, iterator):
        if not hasattr(iterator, '__next__'):
            iterator = iter(iterator)

        while True:
            try:
                with self._profiler():
                    yield next(iterator)
            except StopIteration:
                break
