export NUMSTEPS=100000
export BUFFER_SIZE=20000

python examples/cartpole_ddpg.py --num_steps=$NUMSTEPS --buffer_type=prioritized --buffer_size=$BUFFER_SIZE --seed=1 --savedir='results/cartpole_ddpg/prioritized_buffer_large'
python examples/cartpole_ddpg.py --num_steps=$NUMSTEPS --buffer_type=prioritized --buffer_size=$BUFFER_SIZE --seed=2 --savedir='results/cartpole_ddpg/prioritized_buffer_large'
python examples/cartpole_ddpg.py --num_steps=$NUMSTEPS --buffer_type=prioritized --buffer_size=$BUFFER_SIZE --seed=3 --savedir='results/cartpole_ddpg/prioritized_buffer_large'
python examples/cartpole_ddpg.py --num_steps=$NUMSTEPS --buffer_type=prioritized --buffer_size=$BUFFER_SIZE --seed=4 --savedir='results/cartpole_ddpg/prioritized_buffer_large'
python examples/cartpole_ddpg.py --num_steps=$NUMSTEPS --buffer_type=prioritized --buffer_size=$BUFFER_SIZE --seed=5 --savedir='results/cartpole_ddpg/prioritized_buffer_large'
python examples/cartpole_ddpg.py --num_steps=$NUMSTEPS --buffer_type=prioritized --buffer_size=$BUFFER_SIZE --seed=6 --savedir='results/cartpole_ddpg/prioritized_buffer_large'
python examples/cartpole_ddpg.py --num_steps=$NUMSTEPS --buffer_type=prioritized --buffer_size=$BUFFER_SIZE --seed=7 --savedir='results/cartpole_ddpg/prioritized_buffer_large'
python examples/cartpole_ddpg.py --num_steps=$NUMSTEPS --buffer_type=prioritized --buffer_size=$BUFFER_SIZE --seed=8 --savedir='results/cartpole_ddpg/prioritized_buffer_large'
python examples/cartpole_ddpg.py --num_steps=$NUMSTEPS --buffer_type=prioritized --buffer_size=$BUFFER_SIZE --seed=9 --savedir='results/cartpole_ddpg/prioritized_buffer_large'
python examples/cartpole_ddpg.py --num_steps=$NUMSTEPS --buffer_type=prioritized --buffer_size=$BUFFER_SIZE --seed=10 --savedir='results/cartpole_ddpg/prioritized_buffer_large'

python examples/cartpole_ddpg.py --num_steps=$NUMSTEPS --buffer_type=normal --buffer_size=$BUFFER_SIZE --seed=1 --savedir='results/cartpole_ddpg/normal_buffer_large'
python examples/cartpole_ddpg.py --num_steps=$NUMSTEPS --buffer_type=normal --buffer_size=$BUFFER_SIZE --seed=2 --savedir='results/cartpole_ddpg/normal_buffer_large'
python examples/cartpole_ddpg.py --num_steps=$NUMSTEPS --buffer_type=normal --buffer_size=$BUFFER_SIZE --seed=3 --savedir='results/cartpole_ddpg/normal_buffer_large'
python examples/cartpole_ddpg.py --num_steps=$NUMSTEPS --buffer_type=normal --buffer_size=$BUFFER_SIZE --seed=4 --savedir='results/cartpole_ddpg/normal_buffer_large'
python examples/cartpole_ddpg.py --num_steps=$NUMSTEPS --buffer_type=normal --buffer_size=$BUFFER_SIZE --seed=5 --savedir='results/cartpole_ddpg/normal_buffer_large'
python examples/cartpole_ddpg.py --num_steps=$NUMSTEPS --buffer_type=normal --buffer_size=$BUFFER_SIZE --seed=6 --savedir='results/cartpole_ddpg/normal_buffer_large'
python examples/cartpole_ddpg.py --num_steps=$NUMSTEPS --buffer_type=normal --buffer_size=$BUFFER_SIZE --seed=7 --savedir='results/cartpole_ddpg/normal_buffer_large'
python examples/cartpole_ddpg.py --num_steps=$NUMSTEPS --buffer_type=normal --buffer_size=$BUFFER_SIZE --seed=8 --savedir='results/cartpole_ddpg/normal_buffer_large'
python examples/cartpole_ddpg.py --num_steps=$NUMSTEPS --buffer_type=normal --buffer_size=$BUFFER_SIZE --seed=9 --savedir='results/cartpole_ddpg/normal_buffer_large'
python examples/cartpole_ddpg.py --num_steps=$NUMSTEPS --buffer_type=normal --buffer_size=$BUFFER_SIZE --seed=10 --savedir='results/cartpole_ddpg/normal_buffer_large'

