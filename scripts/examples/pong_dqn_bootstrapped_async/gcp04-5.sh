export SAVEDIR=results/pong_dqn_bootstrapped_async_gcp04/

for SEED in 1 #2 3 4 5 6 7 8 9 10
do
    python3 agentflow/examples/atari_dqn_bootstrapped_async.py \
        --batchsize=128 \
        --begin_learning_at_step=50000 \
        --bootstrap_mask_prob=0.5 \
        --bootstrap_num_heads=64 \
        --bootstrap_prior_scale=2.0 \
        --bootstrap_random_prior=True \
        --bootstrap_explore_before_learning=True \
        --buffer_size=250000 \
        --buffer_type=prioritized \
        --double_q=True \
        --ema_decay=1.0 \
        --target_network_copy_freq=2500 \
        --entropy_loss_weight=0.0 \
        --env_id=PongDeterministic-v4 \
        --learning_rate=2.5e-4 \
        --learning_rate_final=2.5e-4 \
        --learning_rate_decay=0.9999995 \
        --n_envs=100 \
        --n_gpus=1 \
        --n_runners=4 \
        --n_step_return=4 \
        --n_update_steps=1 \
        --network_scale=1 \
        --noise_scale_anneal_steps=1000000 \
        --noise_scale_init=0.4 \
        --noise_scale_final=0.01 \
        --noise_type=eps_greedy \
        --num_steps=10000000 \
        --prioritized_replay_simple=False \
        --ray_port=6007 \
        --weight_decay=0.0 \
        --seed=$SEED \
        --savemodel=True \
        --savedir=$SAVEDIR
done


