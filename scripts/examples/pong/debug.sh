export NUMSTEPS=1000
export BUFFER_SIZE=50000
export BEGIN_LEARNING_AT_STEP=100
export N_UPDATE_STEPS=1
export BATCHSIZE=8
export UPDATE_FREQ=1
export N_PREV_FRAMES=12
export LEARNING_RATE_Q=1e-1
export HIDDEN_DIMS=16
export HIDDEN_LAYERS=2
export N_CPUS=1

export SAVEDIR=results/pong_ddpg_stable_debug/lrq${LEARNING_RATE_Q}_npf${N_PREV_FRAMES}_momentum0.2_nesterov_alpha2_prioritized_init_q5k_wd1e-3_hd${HIDDEN_DIMS}_hl${HIDDEN_LAYERS}

for SEED in 1 #2 3 4 5 6 7 8 9 10
do
    python examples/pong_ddpg_stable.py \
        --n_cpus=$N_CPUS \
        --num_steps=$NUMSTEPS \
        --begin_learning_at_step=$BEGIN_LEARNING_AT_STEP \
        --n_update_steps=$N_UPDATE_STEPS \
        --batchsize=$BATCHSIZE \
        --binarized=True \
        --hidden_dims=$HIDDEN_DIMS \
        --hidden_layers=$HIDDEN_LAYERS \
        --buffer_type=prioritized \
        --prioritized_replay_simple=True \
        --buffer_size=$BUFFER_SIZE \
        --learning_rate_q=$LEARNING_RATE_Q \
        --n_steps_train_only_q 10 \
        --n_prev_frames=$N_PREV_FRAMES \
        --weight_decay=0.001 \
        --optimizer_q=momentum \
        --optimizer_q_momentum=0.2 \
        --alpha=2 \
        --optimizer_q_use_nesterov=True \
        --seed=$SEED \
        --savedir=$SAVEDIR \
        --update_freq=$UPDATE_FREQ \
        --n_steps_per_eval=1000 
done
