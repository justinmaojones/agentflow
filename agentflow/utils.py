from contextlib import contextmanager
import h5py
import os
import time
import yaml
import numpy as np
import pandas as pd
import tensorflow as tf

def check_whats_connected(output):
    for v in tf.global_variables():
        g = tf.gradients([output],v)[0]
        if g is None:
            print('NONE    ',v.name)
        else:
            print('GRADIENT',v.name)

class IdleTimer(object):

    def __init__(self, start_on_create=True):
        self.idle_duration = 0
        self.duration = 0
        self.start_on_create = start_on_create

        self.idle = True 
        self.start_time = None 
        self.prev_time = None

        if self.start_on_create:
            self.start_timer()

    def start_timer(self, idle=True):
        self.idle = idle
        self.start_time = time.time()
        self.reset_timer()

    def reset_timer(self):
        self.prev_time = time.time()

    def __call__(self, idle):
        if self.prev_time is None:
            self.start_timer()
        if self.idle:
            self.idle_duration += time.time() - self.prev_time
        self.duration = time.time() - self.start_time
        self.prev_time = time.time()
        self.idle = idle

    def fraction_idle(self):
        self.__call__(self.idle)
        if self.duration > 0:
            return float(self.idle_duration) / self.duration
        else:
            return 0.

class ScopedIdleTimer(IdleTimer):

    def __init__(self, scope=None, start_on_create=True):
        super(ScopedIdleTimer, self).__init__(start_on_create)
        self._timed_scopes = {}
        self._scopes = []
        assert scope is None or isinstance(scope, str)
        self._scope = scope

    def _add(self, key, duration):
        key = str(key)
        if key not in self._timed_scopes:
            self._timed_scopes[key] = 0
        self._timed_scopes[key] += duration

    @contextmanager
    def time(self, scope):
        assert scope != 'idle'
        assert self.idle
        self.__call__(False)
        start = time.time()
        try:
            yield None
        finally:
            duration = time.time() - start
            self._add(scope, duration)
            self.__call__(True)

    def _get_scoped_key(self, key):
        if self._scope is not None:
            key = self._scope + "/" + key
        return key

    def summary(self):
        self.__call__(self.idle)
        assert self.duration > 0
        output = {}
        for k in self._timed_scopes:
            output[self._get_scoped_key(k)] = float(self._timed_scopes[k]) / self.duration
        output[self._get_scoped_key('idle')] = float(self.idle_duration) / self.duration
        return output

def flatten_logs(logs: dict):
    output = {}
    def _flatten(x, prefix):
        if isinstance(x, dict):
            for k in x:
                prefix.append(k)
                _flatten(x[k], prefix)
                prefix.pop()
        else:
            output['/'.join(prefix)] = x
    _flatten(logs, [])
    return output
    

def load_hdf5(filepath, load_keys=None):
    assert load_keys is None or isinstance(load_keys, list), "load_keys must be None or a list"

    if load_keys:
        with h5py.File(filepath,'r') as f:
            return {k: np.array(f[k]) for k in load_keys}

    else:
        def _load_data(f):
            output = {}
            for k in f:
                if isinstance(f[k], h5py.Group):
                    output[k] = _load_data(f[k])
                else:
                    output[k] = np.array(f[k])
            return output
        with h5py.File(filepath,'r') as f:
            return _load_data(f)

def load_yaml(filepath):
    with open(filepath,'r') as f:
        return yaml.load(f, Loader=yaml.Loader)

def load_experiment_results(savedir, allow_load_partial=False, load_keys=None):
    """
    load results for a single experiment directory
    """
    config_filepath = os.path.join(savedir,'config.yaml')
    if os.path.exists(config_filepath):
        config = load_yaml(os.path.join(savedir,'config.yaml'))
    else:
        config = {}

    log_filepath = os.path.join(savedir,'log.h5')
    log_partial_filepath = os.path.join(savedir,'log_intermediate.h5')

    if os.path.exists(log_filepath):
        logs = load_hdf5(log_filepath, load_keys)

    elif allow_load_partial and os.path.exists(log_partial_filepath):
        logs = load_hdf5(log_partial_filepath, load_keys)

    else:
        logs = {}
        print('could not load logs for %s'%savedir)

    logs = flatten_logs(logs)

    return logs, config

def load_multiple_experiments(resultsdir,allow_load_partial=False,load_empty=False, load_keys=None):
    """
    load results for a multiple experiment directories
    returns a dictionary keyed by the experiment directory 
    """
    savedirs = {k: os.path.join(resultsdir,k) for k in os.listdir(resultsdir)}
    configs = {}
    logs = {}
    for sd in sorted(savedirs):
        if os.path.isdir(savedirs[sd]):
            try:
                log, config = load_experiment_results(savedirs[sd],allow_load_partial,load_keys)
                if len(log) > 0 or load_empty:
                    logs[sd] = log
                    configs[sd] = config
            except BlockingIOError:
                print(f"could not load logs due to BlockingIOError for {savedirs[sd]}")

    return logs, configs

def stack_logs(logs):
    assert isinstance(logs,list)
    keys = list(set([k for l in logs for k in l.keys()]))
    output = {}
    for k in keys:
        logs_k = [l[k] for l in logs]
        assert len(set(map(type,logs_k))) == 1, "log object myst be same type across list"
        if isinstance(logs_k[0],dict):
            output[k] = stack_logs(logs_k)
        else:
            # assume it's an nd.array
            output[k] = np.stack(logs_k)
    return output

def load_multiple_experiments_as_arrays(resultsdir,allow_load_partial=False):
    logs, configs = load_multiple_experiments(resultsdir,allow_load_partial)

    assert set(logs.keys()) == set(configs.keys())

    configs = [configs[k] for k in sorted(configs)]
    logs = [logs[k] for k in sorted(logs)]

    df_configs = pd.DataFrame(configs)
    stacked_logs = stack_logs(logs)

    return stacked_logs, df_configs

def smooth(x,k=10.):
    return np.convolve(x,np.ones(int(k))*1./k)

def plot_percentiles(x,y=None,label=None,smoothing=False,perc_low=10,perc_high=90):
    import matplotlib.pyplot as plt
    if y is None:
        y = x
        x = np.arange(len(y))
    plow = np.percentile(y,perc_low,axis=1)
    mean = y.mean(axis=1)
    phigh = np.percentile(y,perc_high,axis=1)

    if smoothing:
        f = lambda y: smooth(y)
        plow = f(plow)
        mean = f(mean)
        phigh = f(phigh)
        x = np.arange(len(mean))

    plt.fill_between(x,plow,phigh,alpha=0.25)
    plt.plot(x,mean,label=label)

def get_hyperparameters(df_configs,threshold=0.9):
    hyperparameters = {} 
    for c in df_configs.columns:
        vals = df_configs[c].unique().tolist()
        n_unique = len(vals)
        if n_unique < threshold*len(df_configs) and n_unique > 1:
            hyperparameters[c] = vals
    return hyperparameters
