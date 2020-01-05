export NUMSTEPS=100000
export BUFFER_SIZE=20000

python examples/cartpole_ddpg.py --num_steps=$NUMSTEPS --buffer_type=prioritized --buffer_size=$BUFFER_SIZE --add_episode_time_state=True --seed=1 --savedir='results/deleteme/prioritized_buffer_large_add_time_state'
