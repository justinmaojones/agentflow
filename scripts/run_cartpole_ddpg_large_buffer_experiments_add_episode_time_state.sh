export NUMSTEPS=100000
export BUFFER_SIZE=20000

python examples/cartpole_ddpg.py --num_steps=$NUMSTEPS --buffer_type=prioritized --buffer_size=$BUFFER_SIZE --add_episode_time_state=True --seed=1 --savedir='results/cartpole_ddpg/prioritized_buffer_large_add_time_state'
python examples/cartpole_ddpg.py --num_steps=$NUMSTEPS --buffer_type=prioritized --buffer_size=$BUFFER_SIZE --add_episode_time_state=True --seed=2 --savedir='results/cartpole_ddpg/prioritized_buffer_large_add_time_state'
python examples/cartpole_ddpg.py --num_steps=$NUMSTEPS --buffer_type=prioritized --buffer_size=$BUFFER_SIZE --add_episode_time_state=True --seed=3 --savedir='results/cartpole_ddpg/prioritized_buffer_large_add_time_state'
python examples/cartpole_ddpg.py --num_steps=$NUMSTEPS --buffer_type=prioritized --buffer_size=$BUFFER_SIZE --add_episode_time_state=True --seed=4 --savedir='results/cartpole_ddpg/prioritized_buffer_large_add_time_state'
python examples/cartpole_ddpg.py --num_steps=$NUMSTEPS --buffer_type=prioritized --buffer_size=$BUFFER_SIZE --add_episode_time_state=True --seed=5 --savedir='results/cartpole_ddpg/prioritized_buffer_large_add_time_state'
python examples/cartpole_ddpg.py --num_steps=$NUMSTEPS --buffer_type=prioritized --buffer_size=$BUFFER_SIZE --add_episode_time_state=True --seed=6 --savedir='results/cartpole_ddpg/prioritized_buffer_large_add_time_state'
python examples/cartpole_ddpg.py --num_steps=$NUMSTEPS --buffer_type=prioritized --buffer_size=$BUFFER_SIZE --add_episode_time_state=True --seed=7 --savedir='results/cartpole_ddpg/prioritized_buffer_large_add_time_state'
python examples/cartpole_ddpg.py --num_steps=$NUMSTEPS --buffer_type=prioritized --buffer_size=$BUFFER_SIZE --add_episode_time_state=True --seed=8 --savedir='results/cartpole_ddpg/prioritized_buffer_large_add_time_state'
python examples/cartpole_ddpg.py --num_steps=$NUMSTEPS --buffer_type=prioritized --buffer_size=$BUFFER_SIZE --add_episode_time_state=True --seed=9 --savedir='results/cartpole_ddpg/prioritized_buffer_large_add_time_state'
python examples/cartpole_ddpg.py --num_steps=$NUMSTEPS --buffer_type=prioritized --buffer_size=$BUFFER_SIZE --add_episode_time_state=True --seed=10 --savedir='results/cartpole_ddpg/prioritized_buffer_large_add_time_state'
