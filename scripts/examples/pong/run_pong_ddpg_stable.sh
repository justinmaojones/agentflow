export NUMSTEPS=1000000
export BUFFER_SIZE=100000
export BEGIN_LEARNING_AT_STEP=2000
export N_UPDATE_STEPS=16
export BATCHSIZE=32
export UPDATE_FREQ=64
export LEARNING_RATE_Q=10
export BUFFER_TYPE=normal
export NORMALIZE_INPUTS=False
export HIDDEN_DIMS=16

export EXPERIMENT_NAME=${BUFFER_TYPE}_lrq${LEARNING_RATE_Q}_bs${BATCHSIZE}_ni${NORMALIZE_INPUTS}_hd${HIDDEN_DIMS}
export SAVEDIR=results/pong_ddpg_stable/${EXPERIMENT_NAME}

for SEED in 1 #2 3 4 5 6 7 8 9 10
do
    python examples/pong_ddpg_stable.py \
        --num_steps=$NUMSTEPS \
        --begin_learning_at_step=$BEGIN_LEARNING_AT_STEP \
        --n_update_steps=$N_UPDATE_STEPS \
        --batchsize=$BATCHSIZE \
        --buffer_type=${BUFFER_TYPE} \
        --buffer_size=$BUFFER_SIZE \
        --hidden_dims=$HIDDEN_DIMS \
        --seed=$SEED \
        --savedir=$SAVEDIR \
        --update_freq=$UPDATE_FREQ \
        --n_steps_per_eval=1000 \
        --normalize_inputs=$NORMALIZE_INPUTS \
        --learning_rate_q=${LEARNING_RATE_Q}
done
