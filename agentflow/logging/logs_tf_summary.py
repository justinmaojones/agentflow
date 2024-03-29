import h5py
import os
import numpy as np
import tensorflow as tf
from typing import Dict, List, Union


class LogsTFSummary:
    def __init__(self, savedir: str, filename: str = "log.h5", **kwargs):
        self.logs = {}
        self.savedir = savedir
        self.kwargs = kwargs
        self._summary_writer = None
        self._log_filepath = os.path.join(self.savedir, filename)
        self._step = None

    def __getitem__(self, key: str):
        if key not in self.logs:
            self.logs[key] = []
        return self.logs[key]

    @property
    def summary_writer(self):
        # lazy build of summary writer
        if self._summary_writer is None:
            self._summary_writer = tf.summary.create_file_writer(
                self.savedir, **self.kwargs
            )
        return self._summary_writer

    def append(self, key: str, val: Union[float, int, np.ndarray]):
        val = np.mean(val)

        # logs are stored and later flushed into an hdf5 file
        if key not in self.logs:
            self.logs[key] = []
        self.logs[key].append(val)

        # WARNING: tf.summary overwrites previously written values
        # for the same step
        with self.summary_writer.as_default():
            tf.summary.scalar(key, val, step=self.step)

    def extend(self, key: str, vals: List[Union[float, int, np.ndarray]]):
        # dont just list.extend, since we want to capture tf.summary
        for val in vals:
            self.append(key, val)

    def append_dict(self, inp: Dict[str, Union[float, int, np.ndarray]]):
        for k in inp:
            if isinstance(inp[k], dict):
                v = {f"{k}/{k2}": inp[k][k2] for k2 in inp[k]}
                self.append_dict(v)
            else:
                self.append(k, inp[k])

    def set_step(self, t: int):
        tf.summary.experimental.set_step(t)
        self._step = t

    @property
    def step(self):
        return self._step

    def stack(self, key: str = None):
        if key is None:
            return {key: self.stack(key) for key in self.logs}
        else:
            return np.stack(self.logs[key])

    def flush(self):
        self.summary_writer.flush()
        self._write_h5()
        self.logs = {}

    def with_filename(self, filename: str):
        return LogsTFSummary(savedir=self.savedir, filename=filename, **self.kwargs)

    def _write_h5(self):
        with h5py.File(self._log_filepath, "a") as f:
            for key in sorted(self.logs):
                data = np.array(self.logs[key])
                try:
                    if key not in f:
                        dataset = f.create_dataset(
                            key,
                            data.shape,
                            dtype=data.dtype,
                            chunks=data.shape,
                            maxshape=tuple([None] + list(data.shape[1:])),
                        )
                        dataset[:] = data

                    else:
                        dataset = f[key]
                        n = len(f[key])
                        m = len(data)
                        f[key].resize(n + m, axis=0)
                        f[key][n:] = data
                except TypeError:
                    for k in self.logs:
                        if k in key and k != key:
                            raise TypeError(
                                f"cannot have parent='{k}' and child='{key}' log paths in same dataset"
                            )
                        if key in k and k != key:
                            raise TypeError(
                                f"cannot have parent='{key}' and child='{k}' log paths in same dataset"
                            )
