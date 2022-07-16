#!/bin/sh

export SAVEDIR=results/pendulum_ddpg/run

python agentflow/examples/pendulum_ddpg.py \
    --n_envs=10 \
    --batchsize=100 \
    --ema_decay=0.99 \
    --batchnorm=True \
    --noise_eps=0.5 \
    --num_steps=30000 \
    --n_prev_frames=16 \
    --begin_learning_at_step=1000 \
    --n_update_steps=4 \
    --update_freq=4 \
    --enable_n_step_return_publisher=True \
    --n_step_return=4 \
    --prioritized_replay_simple=False \
    --buffer_type=normal \
    --hidden_dims=64 \
    --hidden_layers=2 \
    --buffer_size=30000 \
    --learning_rate=1e-3 \
    --learning_rate_decay=0.9999 \
    --policy_loss_weight=10.0 \
    --weight_decay=1e-4 \
    --seed=1 \
    --savedir=$SAVEDIR 
