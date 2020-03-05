export NUMSTEPS=1000000
export BUFFER_SIZE=100000
export BEGIN_LEARNING_AT_STEP=10000
export N_UPDATE_STEPS=16
export BATCHSIZE=64
export UPDATE_FREQ=64
export LEARNING_RATE_Q=100
export HIDDEN_DIMS=32

export SAVEDIR="results/pong_ddpg_stable/normal_lrq${LEARNING_RATE_Q}_bs${BATCHSIZE}_hd${HIDDEN_DIMS}"

for SEED in 1 #2 3 4 5 6 7 8 9 10
do
    python examples/pong_ddpg_stable.py \
        --num_steps=$NUMSTEPS \
        --begin_learning_at_step=$BEGIN_LEARNING_AT_STEP \
        --n_update_steps=$N_UPDATE_STEPS \
        --batchsize=$BATCHSIZE \
        --buffer_type=normal \
        --buffer_size=$BUFFER_SIZE \
        --seed=$SEED \
        --savedir=$SAVEDIR \
        --update_freq=$UPDATE_FREQ \
        --n_steps_per_eval=1000 \
        --learning_rate_q=$LEARNING_RATE_Q \
        --hidden_dims=$HIDDEN_DIMS
done
