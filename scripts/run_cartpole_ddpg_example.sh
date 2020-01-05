export NUMSTEPS=20000
export BEGIN_LEARNING_AT_STEP=10000

python examples/cartpole_ddpg.py --num_steps=$NUMSTEPS --begin_learning_at_step=$BEGIN_LEARNING_AT_STEP --buffer_type=prioritized --seed=1 --savedir='results/cartpole_ddpg/prioritized_buffer'
#python examples/cartpole_ddpg.py --num_steps=$NUMSTEPS --begin_learning_at_step=$BEGIN_LEARNING_AT_STEP --buffer_type=normal --seed=1 --savedir='results/cartpole_ddpg/normal_buffer'
