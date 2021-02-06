export SAVEDIR=results/pong_dqn_bootstrapped_async_gcp07/

for SEED in 1 #2 3 4 5 6 7 8 9 10
do
    python3 agentflow/examples/atari_dqn_bootstrapped_async.py \
        --batchsize=32 \
        --begin_learning_at_step=320000 \
        --bootstrap_mask_prob=1.0 \
        --bootstrap_num_heads=1 \
        --bootstrap_prior_scale=0.0 \
        --bootstrap_random_prior=False \
        --bootstrap_explore_before_learning=False \
        --buffer_size=320000 \
        --buffer_type=normal \
        --dueling=True \
        --double_q=True \
        --ema_decay=1.0 \
        --target_network_copy_freq=250 \
        --env_id=PongNoFrameskip-v4 \
        --grad_clip_norm=10.0 \
        --learning_rate=1e-4 \
        --learning_rate_final=1e-4 \
        --learning_rate_decay=0.9999995 \
        --log_flush_freq=1000 \
        --n_envs=8 \
        --n_steps_per_eval=10000 \
        --n_gpus=1 \
        --n_runners=4 \
        --n_step_return=1 \
        --n_update_steps=1 \
        --network_scale=2 \
        --noise_scale_anneal_steps=250000 \
        --noise_scale_init=1.0 \
        --noise_scale_final=0.01 \
        --noise_type=eps_greedy \
        --normalize_inputs=True \
        --num_steps=10000000 \
        --prioritized_replay_simple=False \
        --runner_update_freq=8 \
        --ray_port=6007 \
        --td_loss=huber \
        --weight_decay=0.0 \
        --seed=$SEED \
        --savemodel=False \
        --savedir=$SAVEDIR
done


