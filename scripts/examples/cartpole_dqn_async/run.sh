#!/bin/sh

export SAVEDIR=results/cartpole_dqn_async/run

python agentflow/examples/cartpole_dqn_async.py \
    --runner_threads=4 \
    --parameter_server_cpu=4 \
    --parameter_server_threads=4 \
    --n_envs=16 \
    --batchsize=2048 \
    --dataset_prefetch=8 \
    --min_parallel_sample_rpc=8 \
    --ema_decay=0.999 \
    --num_steps=20000 \
    --noise_eps=0.5 \
    --begin_learning_at_step=200 \
    --n_updates_per_model_refresh=32 \
    --n_step_return=8 \
    --buffer_size=30000 \
    --learning_rate=3e-3 \
    --learning_rate_decay=0.9999 \
    --weight_decay=1e-4 \
    --savedir=$SAVEDIR
