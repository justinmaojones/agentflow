import h5py
import os
import numpy as np
import tensorflow as tf

class LogsTFSummary(object):

    def __init__(self,savedir,**kwargs):
        self.logs = {}
        self.savedir = savedir
        self.summary_writer = tf.summary.create_file_writer(savedir,**kwargs)
        self._log_filepath = os.path.join(self.savedir,'log.h5')

    def __getitem__(self, key):
        if key not in self.logs:
            self.logs[key] = []
        return self.logs[key]

    def _append(self, key, val):
        if key not in self.logs:
            self.logs[key] = []
        self.logs[key].append(val)
        with self.summary_writer.as_default():
            tf.summary.scalar(key, np.mean(val))

    def append(self, key, val, summary_only=True):
        # TODO: very hacky converting tensors to numpy
        if isinstance(val, tf.Tensor):
            val = val.numpy()
        if summary_only:
            self._append(key, np.mean(val))
        else:
            self._append(key, val)

    def append_seq(self, key, vals):
        for i in range(len(vals)):
            self.append(key, vals[i])

    def append_dict(self, inp, summary_only=True):
        for k in inp:
            if isinstance(inp[k], dict):
                v = {f"{k}/{k2}":inp[k][k2] for k2 in inp[k]}
                self.append_dict(v, summary_only)
            else:
                self.append(k, inp[k], summary_only)

    def set_step(self, t):
        tf.summary.experimental.set_step(t)

    def stack(self,key=None):
        if key is None:
            return {key:self.stack(key) for key in self.logs}
        else:
            return np.stack(self.logs[key])

    def flush(self, verbose=False):
        self.summary_writer.flush()
        self.write(self._log_filepath, verbose)
        self.logs = {}

    def with_prefix(self, prefix):
        return LogsTFSummaryFlow(self, prefix) 

    def write(self, filepath, verbose=True):
        if verbose:
            print(f"Writing h5 file to: {filepath}")
        with h5py.File(filepath, 'a') as f:
            for key in sorted(self.logs):
                data = np.array(self.logs[key])
                key = key.replace('/','_')
                if verbose:
                    print('H5: %s %s'%(key,str(data.shape)))
                if key not in f:
                    dataset = f.create_dataset(
                        key, 
                        data.shape, 
                        dtype=data.dtype,
                        chunks=data.shape,
                        maxshape=tuple([None]+list(data.shape[1:]))
                    )
                    dataset[:] = data

                else:
                    dataset = f[key]
                    n = len(f[key])
                    m = len(data)
                    f[key].resize(n + m,axis=0)
                    f[key][n:] = data

class LogsTFSummaryFlow(LogsTFSummary):

    def __init__(self, log: LogsTFSummary, prefix: str):
        self.log = log
        self.prefix = prefix

    def _prefixed_key(self, key):
        return f"{self.prefix}/{key}"

    def __getitem__(self, key):
        return self.log[self._prefixed_key(key)]

    def _append(self, key, val):
        self.log._append(self._prefixed_key(key), val)

    def append(self, key, val, summary_only=True):
        self.log.append(self._prefixed_key(key), val, summary_only)

    def set_step(self, t):
        self.log.set_step(t)

    def stack(self, key=None):
        if key is not None:
            key = self._prefixed_key(key)
        return self.log.stack(key)

    def flush(self, verbose=False):
        self.log.flush(verbose)

    def write(self, filepath, verbose=True):
        self.log.write(filepath, verbose)
