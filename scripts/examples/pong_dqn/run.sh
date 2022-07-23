#!/bin/sh

export SAVEDIR=results/atari_dqn/pong/run
python agentflow/examples/atari_dqn.py \
    --batchsize=32 \
    --begin_learning_at_step=100000 \
    --buffer_size=100000 \
    --frames_per_action=4 \
    --learning_rate=1e-4 \
    --learning_rate_decay=0.999999999999 \
    --n_envs=1 \
    --n_prev_frames=4 \
    --n_step_return=1 \
    --n_update_steps=1 \
    --network_scale=2 \
    --noise_scale_init=1.0 \
    --noise_scale_final=0.01 \
    --noise_scale_anneal_steps=1000000 \
    --num_steps=2000000 \
    --savedir=$SAVEDIR \
    --seed=1 \
    --update_freq=4 
