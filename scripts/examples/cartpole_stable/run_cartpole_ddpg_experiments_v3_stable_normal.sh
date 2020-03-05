export NUMSTEPS=10000
export BUFFER_SIZE=10000
export BEGIN_LEARNING_AT_STEP=100
export N_UPDATE_STEPS=4

export SAVEDIR=results/cartpole_ddpg_stable/normal

for SEED in 1 2 3 4 5 6 7 8 9 10
do
    python examples/cartpole_ddpg_stable.py \
        --num_steps=$NUMSTEPS \
        --begin_learning_at_step=$BEGIN_LEARNING_AT_STEP \
        --n_update_steps=$N_UPDATE_STEPS \
        --buffer_type=normal \
        --buffer_size=$BUFFER_SIZE \
        --add_episode_time_state=True \
        --seed=$SEED \
        --savedir=$SAVEDIR
done
