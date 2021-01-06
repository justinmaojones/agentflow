export SAVEDIR=results/pong_dqn_bootstrapped_gcp01/

for SEED in 1 #2 3 4 5 6 7 8 9 10
do
    python agentflow/examples/atari_dqn_bootstrapped.py \
        --batchsize=64 \
        --begin_learning_at_step=50000 \
        --bootstrap_mask_prob=0.5 \
        --bootstrap_num_heads=64 \
        --bootstrap_prior_scale=2.0 \
        --bootstrap_random_prior=True \
        --buffer_size=300000 \
        --buffer_type=normal \
        --double_q=True \
        --ema_decay=0.95 \
        --enable_n_step_return_publisher=True \
        --entropy_loss_weight=0.0 \
        --env_id=PongDeterministic-v4 \
        --learning_rate=1e-4 \
        --learning_rate_decay=0.99995 \
        --n_envs=8 \
        --n_update_steps=1 \
        --n_step_return=32 \
        --network_scale=1 \
        --noise=eps_greedy \
        --noise_eps=0.01 \
        --num_steps=10000000 \
        --prioritized_replay_simple=True \
        --weight_decay=1e-4 \
        --seed=$SEED \
        --savedir=$SAVEDIR
done


