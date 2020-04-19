export NUMSTEPS=10000
export BUFFER_SIZE=10000
export BEGIN_LEARNING_AT_STEP=100
export N_UPDATE_STEPS=4

export SAVEDIR=results/cartpole_ddpg_stable/lrq1e-1_momentum0.2_nesterov_alpha2_prioritized_binarized

for SEED in 1 2 3 4 5 6 7 8 9 10
do
    python examples/cartpole_ddpg_stable.py \
        --num_steps=$NUMSTEPS \
        --begin_learning_at_step=$BEGIN_LEARNING_AT_STEP \
        --n_update_steps=$N_UPDATE_STEPS \
        --buffer_type=prioritized \
        --buffer_size=$BUFFER_SIZE \
        --prioritized_replay_simple=True \
        --add_episode_time_state=True \
        --binarized_time_state=True \
        --learning_rate_q=1e-1 \
        --optimizer_q=momentum \
        --optimizer_q_momentum=0.2 \
        --alpha=2 \
        --optimizer_q_use_nesterov=True \
        --seed=$SEED \
        --savedir=$SAVEDIR
done


