#!/bin/sh

export SAVEDIR=results/atari_dqn_async/pong/run

python agentflow/examples/atari_dqn_async.py \
    --runner_count=1 \
    --runner_threads=4 \
    --parameter_server_cpu=8 \
    --parameter_server_threads=4 \
    --batchsize=32 \
    --begin_learning_at_step=100000 \
    --buffer_size=100000 \
    --dataset_prefetch=8 \
    --min_parallel_sample_rpc=8 \
    --ema_decay=0.99 \
    --frames_per_action=4 \
    --learning_rate=1e-4 \
    --learning_rate_decay=0.999999999999 \
    --n_envs=1 \
    --n_prev_frames=4 \
    --n_step_return=1 \
    --n_updates_per_model_refresh=8 \
    --network_scale=2 \
    --noise_scale_init=1.0 \
    --noise_scale_final=0.01 \
    --noise_scale_anneal_steps=250000 \
    --num_steps=2000000 \
    --savedir=$SAVEDIR \
    --seed=1 \
    --weight_decay=0 \
