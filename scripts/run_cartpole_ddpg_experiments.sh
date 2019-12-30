export NUMSTEPS=200

python examples/cartpole_ddpg.py --num_steps=$NUMSTEPS --buffer_type=prioritized --seed=1 --savedir='results/cartpole_ddpg/prioritized_buffer'
python examples/cartpole_ddpg.py --num_steps=$NUMSTEPS --buffer_type=prioritized --seed=2 --savedir='results/cartpole_ddpg/prioritized_buffer'
python examples/cartpole_ddpg.py --num_steps=$NUMSTEPS --buffer_type=prioritized --seed=3 --savedir='results/cartpole_ddpg/prioritized_buffer'
python examples/cartpole_ddpg.py --num_steps=$NUMSTEPS --buffer_type=prioritized --seed=4 --savedir='results/cartpole_ddpg/prioritized_buffer'
python examples/cartpole_ddpg.py --num_steps=$NUMSTEPS --buffer_type=prioritized --seed=5 --savedir='results/cartpole_ddpg/prioritized_buffer'
python examples/cartpole_ddpg.py --num_steps=$NUMSTEPS --buffer_type=prioritized --seed=6 --savedir='results/cartpole_ddpg/prioritized_buffer'
python examples/cartpole_ddpg.py --num_steps=$NUMSTEPS --buffer_type=prioritized --seed=7 --savedir='results/cartpole_ddpg/prioritized_buffer'
python examples/cartpole_ddpg.py --num_steps=$NUMSTEPS --buffer_type=prioritized --seed=8 --savedir='results/cartpole_ddpg/prioritized_buffer'
python examples/cartpole_ddpg.py --num_steps=$NUMSTEPS --buffer_type=prioritized --seed=9 --savedir='results/cartpole_ddpg/prioritized_buffer'
python examples/cartpole_ddpg.py --num_steps=$NUMSTEPS --buffer_type=prioritized --seed=10 --savedir='results/cartpole_ddpg/prioritized_buffer'

python examples/cartpole_ddpg.py --num_steps=$NUMSTEPS --buffer_type=normal --seed=1 --savedir='results/cartpole_ddpg/normal_buffer'
python examples/cartpole_ddpg.py --num_steps=$NUMSTEPS --buffer_type=normal --seed=2 --savedir='results/cartpole_ddpg/normal_buffer'
python examples/cartpole_ddpg.py --num_steps=$NUMSTEPS --buffer_type=normal --seed=3 --savedir='results/cartpole_ddpg/normal_buffer'
python examples/cartpole_ddpg.py --num_steps=$NUMSTEPS --buffer_type=normal --seed=4 --savedir='results/cartpole_ddpg/normal_buffer'
python examples/cartpole_ddpg.py --num_steps=$NUMSTEPS --buffer_type=normal --seed=5 --savedir='results/cartpole_ddpg/normal_buffer'
python examples/cartpole_ddpg.py --num_steps=$NUMSTEPS --buffer_type=normal --seed=6 --savedir='results/cartpole_ddpg/normal_buffer'
python examples/cartpole_ddpg.py --num_steps=$NUMSTEPS --buffer_type=normal --seed=7 --savedir='results/cartpole_ddpg/normal_buffer'
python examples/cartpole_ddpg.py --num_steps=$NUMSTEPS --buffer_type=normal --seed=8 --savedir='results/cartpole_ddpg/normal_buffer'
python examples/cartpole_ddpg.py --num_steps=$NUMSTEPS --buffer_type=normal --seed=9 --savedir='results/cartpole_ddpg/normal_buffer'
python examples/cartpole_ddpg.py --num_steps=$NUMSTEPS --buffer_type=normal --seed=10 --savedir='results/cartpole_ddpg/normal_buffer'

