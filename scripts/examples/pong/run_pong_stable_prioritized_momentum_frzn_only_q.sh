export NUMSTEPS=1000000
export BUFFER_SIZE=10000
export BEGIN_LEARNING_AT_STEP=10000
export N_UPDATE_STEPS=4
export BATCHSIZE=32
export UPDATE_FREQ=64
export N_PREV_FRAMES=12
export LEARNING_RATE_Q=1e-1

export SAVEDIR=results/pong_ddpg_stable/lrq${LEARNING_RATE_Q}_npf${N_PREV_FRAMES}_momentum0.2_nesterov_alpha2_frzn_prioritized_only_q_10k

for SEED in 1 #2 3 4 5 6 7 8 9 10
do
    python examples/pong_ddpg_stable.py \
        --num_steps=$NUMSTEPS \
        --begin_learning_at_step=$BEGIN_LEARNING_AT_STEP \
        --n_update_steps=$N_UPDATE_STEPS \
        --batchsize=$BATCHSIZE \
        --buffer_type=prioritized \
        --prioritized_replay_simple=True \
        --buffer_size=$BUFFER_SIZE \
        --learning_rate_q=$LEARNING_RATE_Q \
        --n_steps_train_only_q 10000 \
        --n_prev_frames=$N_PREV_FRAMES \
        --freeze_conv_net=True \
        --optimizer_q=momentum \
        --optimizer_q_momentum=0.2 \
        --alpha=2 \
        --optimizer_q_use_nesterov=True \
        --seed=$SEED \
        --savedir=$SAVEDIR \
        --update_freq=$UPDATE_FREQ \
        --n_steps_per_eval=1000 
done
