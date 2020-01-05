export NUMSTEPS=100000
export BUFFER_SIZE=20000

python examples/cartpole_ddpg.py --num_steps=$NUMSTEPS --buffer_type=prioritized --buffer_size=$BUFFER_SIZE --ema_decay=0.9 --prioritized_replay_weights_uniform=True --seed=1 --savedir='results/cartpole_ddpg/prioritized_buffer_large_ema0.9_uniform'
python examples/cartpole_ddpg.py --num_steps=$NUMSTEPS --buffer_type=prioritized --buffer_size=$BUFFER_SIZE --ema_decay=0.9 --prioritized_replay_weights_uniform=True --seed=2 --savedir='results/cartpole_ddpg/prioritized_buffer_large_ema0.9_uniform'
python examples/cartpole_ddpg.py --num_steps=$NUMSTEPS --buffer_type=prioritized --buffer_size=$BUFFER_SIZE --ema_decay=0.9 --prioritized_replay_weights_uniform=True --seed=3 --savedir='results/cartpole_ddpg/prioritized_buffer_large_ema0.9_uniform'
python examples/cartpole_ddpg.py --num_steps=$NUMSTEPS --buffer_type=prioritized --buffer_size=$BUFFER_SIZE --ema_decay=0.9 --prioritized_replay_weights_uniform=True --seed=4 --savedir='results/cartpole_ddpg/prioritized_buffer_large_ema0.9_uniform'
python examples/cartpole_ddpg.py --num_steps=$NUMSTEPS --buffer_type=prioritized --buffer_size=$BUFFER_SIZE --ema_decay=0.9 --prioritized_replay_weights_uniform=True --seed=5 --savedir='results/cartpole_ddpg/prioritized_buffer_large_ema0.9_uniform'
python examples/cartpole_ddpg.py --num_steps=$NUMSTEPS --buffer_type=prioritized --buffer_size=$BUFFER_SIZE --ema_decay=0.9 --prioritized_replay_weights_uniform=True --seed=6 --savedir='results/cartpole_ddpg/prioritized_buffer_large_ema0.9_uniform'
python examples/cartpole_ddpg.py --num_steps=$NUMSTEPS --buffer_type=prioritized --buffer_size=$BUFFER_SIZE --ema_decay=0.9 --prioritized_replay_weights_uniform=True --seed=7 --savedir='results/cartpole_ddpg/prioritized_buffer_large_ema0.9_uniform'
python examples/cartpole_ddpg.py --num_steps=$NUMSTEPS --buffer_type=prioritized --buffer_size=$BUFFER_SIZE --ema_decay=0.9 --prioritized_replay_weights_uniform=True --seed=8 --savedir='results/cartpole_ddpg/prioritized_buffer_large_ema0.9_uniform'
python examples/cartpole_ddpg.py --num_steps=$NUMSTEPS --buffer_type=prioritized --buffer_size=$BUFFER_SIZE --ema_decay=0.9 --prioritized_replay_weights_uniform=True --seed=9 --savedir='results/cartpole_ddpg/prioritized_buffer_large_ema0.9_uniform'
python examples/cartpole_ddpg.py --num_steps=$NUMSTEPS --buffer_type=prioritized --buffer_size=$BUFFER_SIZE --ema_decay=0.9 --prioritized_replay_weights_uniform=True --seed=10 --savedir='results/cartpole_ddpg/prioritized_buffer_large_ema0.9_uniform'

