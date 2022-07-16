from agentflow.examples.grid_search import run_grid_search

NUM_TRIALS = 10
NUM_WORKERS = 5 
RUN_FILE = 'agentflow/examples/cartpole_dqn_bootstrapped.py'
SAVE_DIR = 'results/cartpole_dqn_bootstrapped/run_grid_search'
KWARGS = {
    'bootstrap_num_heads': [64],
    'bootstrap_mask_prob': [0.25, 0.5, 0.75],
    'bootstrap_random_prior': [False, True],
    'batchnorm': [False],
    'batchsize': [64],
    'begin_learning_at_step': [200],
    'buffer_size': [30000],
    'buffer_type': ['normal', 'prioritized'],
    'ema_decay': [0.99],
    'enable_n_step_return_publisher': [True],
    'double_q': [True,False],
    'hidden_dims': [64],
    'hidden_layers': [4],
    'learning_rate': [1e-4],
    'learning_rate_decay': [0.9999],
    'n_envs': [16],
    'n_prev_frames': [16],
    'n_step_return': [4, 8],
    'n_update_steps': [4, 8],
    'noise_eps': [0.5],
    'normalize_inputs': [True],
    'num_steps': [10000],
    'prioritized_replay_simple': [True],
    'savedir': [SAVE_DIR],
    'seed': list(range(10000)),
    'weight_decay': [1e-4],
}

run_grid_search(NUM_TRIALS, NUM_WORKERS, RUN_FILE, SAVE_DIR, KWARGS)
