import h5py
import os
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


class LogsTFSummary(object):

    def __init__(self,savedir,**kwargs):
        self.logs = {}
        self.savedir = savedir
        self.summary_writer = tf.summary.FileWriter(savedir,**kwargs)
        self.summary = tf.Summary()
        self._other_array_metrics = {
            'min': np.min,
            'max': np.max,
            'l2norm': lambda x: np.sqrt(np.sum(np.square(x))),
            'max-min': lambda x: np.max(x.astype(float)) - np.min(x.astype(float)),
        }

    def __getitem__(self, key):
        if key not in self.logs:
            self.logs[key] = []
        return self.logs[key]

    def _append(self,key,val):
        if key not in self.logs:
            self.logs[key] = []
        self.logs[key].append(val)
        self.summary.value.add(
                tag=key,
                simple_value=np.mean(val))

    def append(self,key,val):
        self._append(key,val)
        if np.size(val) > 1:
            for m in self._other_array_metrics:
                k2 = key + '/' + m
                v2 = self._other_array_metrics[m](val)
                self._append(k2,v2)

    def append_dict(self,inp):
        for k in inp:
            self.append(k,inp[k])

    def stack(self,key=None):
        if key is None:
            return {key:self.stack(key) for key in self.logs}
        else:
            return np.stack(self.logs[key])

    def flush(self, step=None):
        self.summary_writer.add_summary(self.summary, step)
        self.summary_writer.flush()
        self.summary = tf.Summary()

    def write(self, filepath, verbose=True):
        if verbose:
            print('WRITING H5 FILE TO: {filepath}'.format(**locals()))
        with h5py.File(filepath, 'w') as f:
            for key in sorted(self.logs):
                if verbose:
                    print('H5: %s %s'%(key,type(self.logs[key])))
                try:
                    f[key] = self.logs[key]
                except:
                    try:
                        f[key] = np.concatenate(self.logs[key])
                    except:
                        pass

def load_hdf5(filepath):
    def load_data(f):
        output = {}
        for k in f:
            if isinstance(f[k],h5py.Group):
                output[k] = load_data(f[k])
            else:
                output[k] = np.array(f[k])
        return output
    with h5py.File(filepath,'r') as f:
        return load_data(f)

def load_yaml(filepath):
    with open(filepath,'r') as f:
        return yaml.load(f)

def load_experiment_results(savedir,allow_load_partial=False):
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
        logs = load_hdf5(log_filepath)

    elif allow_load_partial and os.path.exists(log_partial_filepath):
        logs = load_hdf5(log_partial_filepath)

    else:
        logs = {}
        print('could not load logs for %s'%savedir)

    return logs, config

def load_multiple_experiments(resultsdir,allow_load_partial=False,load_empty=False):
    """
    load results for a multiple experiment directories
    returns a dictionary keyed by the experiment directory 
    """
    savedirs = {k: os.path.join(resultsdir,k) for k in os.listdir(resultsdir)}
    configs = {}
    logs = {}
    for sd in sorted(savedirs):
        if os.path.isdir(savedirs[sd]):
            log, config = load_experiment_results(savedirs[sd],allow_load_partial)
            if len(log) > 0 or load_empty:
                logs[sd] = log
                configs[sd] = config

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

def plot_percentiles(x,y=None,label=None,smoothing=False):
    import matplotlib.pyplot as plt
    if y is None:
        y = x
        x = np.arange(len(y))
    p10 = np.percentile(y,10,axis=1)
    mean = y.mean(axis=1)
    p90 = np.percentile(y,90,axis=1)

    if smoothing:
        f = lambda y: smooth(y)
        p10 = f(p10)
        mean = f(mean)
        p90 = f(p90)
        x = np.arange(len(mean))

    plt.fill_between(x,p10,p90,alpha=0.25)
    plt.plot(x,mean,label=label)

def get_hyperparameters(df_configs,threshold=0.9):
    hyperparameters = {} 
    for c in df_configs.columns:
        vals = df_configs[c].unique().tolist()
        n_unique = len(vals)
        if n_unique < threshold*len(df_configs) and n_unique > 1:
            hyperparameters[c] = vals
    return hyperparameters
