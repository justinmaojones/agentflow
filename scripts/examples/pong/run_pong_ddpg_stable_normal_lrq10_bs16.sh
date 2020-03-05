export NUMSTEPS=1000000
export BUFFER_SIZE=100000
export BEGIN_LEARNING_AT_STEP=10000
export N_UPDATE_STEPS=16
export BATCHSIZE=32
export UPDATE_FREQ=64

export SAVEDIR=results/pong_ddpg_stable/normal_lrq10_bs16

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
        --learning_rate_q=10
done
