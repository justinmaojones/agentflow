export NUMSTEPS=1000000
export BUFFER_SIZE=20000
export BEGIN_LEARNING_AT_STEP=10000
export N_UPDATE_STEPS=1
export BATCHSIZE=64
export UPDATE_FREQ=1
export N_PREV_FRAMES=16
export LEARNING_RATE_Q=1e-1
export N_ENVS=10
export HIDDEN_DIMS=64
export HIDDEN_LAYERS=3

export SAVEDIR=results/pong_ddpg_stable/lrq${LEARNING_RATE_Q}_npf${N_PREV_FRAMES}_momentum0.2_nesterov_alpha2_prioritized_wd1e-4_nenv${N_ENVS}_dims${HIDDEN_DIMS}_lyrs${HIDDEN_LAYERS}

for SEED in 1 #2 3 4 5 6 7 8 9 10
do
    python examples/pong_ddpg_stable.py \
        --num_steps=$NUMSTEPS \
        --n_envs $N_ENVS \
        --begin_learning_at_step=$BEGIN_LEARNING_AT_STEP \
        --n_update_steps=$N_UPDATE_STEPS \
        --batchsize=$BATCHSIZE \
        --buffer_type=prioritized \
        --prioritized_replay_simple=True \
        --buffer_size=$BUFFER_SIZE \
        --learning_rate=0.001 \
        --learning_rate_decay=0.9999 \
        --learning_rate_q=$LEARNING_RATE_Q \
        --learning_rate_q_decay=0.99995 \
        --n_steps_train_only_q 0 \
        --n_prev_frames=$N_PREV_FRAMES \
        --freeze_conv_net=False \
        --hidden_dims=$HIDDEN_DIMS \
        --hidden_layers=$HIDDEN_LAYERS \
        --weight_decay=0.0001 \
        --ema_decay=0.99 \
        --regularize_policy=False \
        --optimizer_q=momentum \
        --optimizer_q_momentum=0.05 \
        --alpha=2 \
        --optimizer_q_use_nesterov=True \
        --seed=$SEED \
        --savedir=$SAVEDIR \
        --update_freq=$UPDATE_FREQ \
        --n_steps_per_eval=100 
done
