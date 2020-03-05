export NUMSTEPS=10000
export BUFFER_SIZE=10000
export BEGIN_LEARNING_AT_STEP=2
export N_UPDATE_STEPS=16

export SAVEDIR=results/cartpole_ddpg/normal_v2

#python examples/cartpole_ddpg.py --num_steps=$NUMSTEPS --begin_learning_at_step=$BEGIN_LEARNING_AT_STEP --n_update_steps=$N_UPDATE_STEPS --buffer_type=normal --buffer_size=$BUFFER_SIZE --add_episode_time_state=True --prioritized_replay_compute_init=True --seed=1 --savedir=$SAVEDIR
#python examples/cartpole_ddpg.py --num_steps=$NUMSTEPS --begin_learning_at_step=$BEGIN_LEARNING_AT_STEP --n_update_steps=$N_UPDATE_STEPS --buffer_type=normal --buffer_size=$BUFFER_SIZE --add_episode_time_state=True --prioritized_replay_compute_init=True --seed=2 --savedir=$SAVEDIR
#python examples/cartpole_ddpg.py --num_steps=$NUMSTEPS --begin_learning_at_step=$BEGIN_LEARNING_AT_STEP --n_update_steps=$N_UPDATE_STEPS --buffer_type=normal --buffer_size=$BUFFER_SIZE --add_episode_time_state=True --prioritized_replay_compute_init=True --seed=3 --savedir=$SAVEDIR
#python examples/cartpole_ddpg.py --num_steps=$NUMSTEPS --begin_learning_at_step=$BEGIN_LEARNING_AT_STEP --n_update_steps=$N_UPDATE_STEPS --buffer_type=normal --buffer_size=$BUFFER_SIZE --add_episode_time_state=True --prioritized_replay_compute_init=True --seed=4 --savedir=$SAVEDIR
#python examples/cartpole_ddpg.py --num_steps=$NUMSTEPS --begin_learning_at_step=$BEGIN_LEARNING_AT_STEP --n_update_steps=$N_UPDATE_STEPS --buffer_type=normal --buffer_size=$BUFFER_SIZE --add_episode_time_state=True --prioritized_replay_compute_init=True --seed=5 --savedir=$SAVEDIR
#python examples/cartpole_ddpg.py --num_steps=$NUMSTEPS --begin_learning_at_step=$BEGIN_LEARNING_AT_STEP --n_update_steps=$N_UPDATE_STEPS --buffer_type=normal --buffer_size=$BUFFER_SIZE --add_episode_time_state=True --prioritized_replay_compute_init=True --seed=6 --savedir=$SAVEDIR
#python examples/cartpole_ddpg.py --num_steps=$NUMSTEPS --begin_learning_at_step=$BEGIN_LEARNING_AT_STEP --n_update_steps=$N_UPDATE_STEPS --buffer_type=normal --buffer_size=$BUFFER_SIZE --add_episode_time_state=True --prioritized_replay_compute_init=True --seed=7 --savedir=$SAVEDIR
#python examples/cartpole_ddpg.py --num_steps=$NUMSTEPS --begin_learning_at_step=$BEGIN_LEARNING_AT_STEP --n_update_steps=$N_UPDATE_STEPS --buffer_type=normal --buffer_size=$BUFFER_SIZE --add_episode_time_state=True --prioritized_replay_compute_init=True --seed=8 --savedir=$SAVEDIR
python examples/cartpole_ddpg.py --num_steps=$NUMSTEPS --begin_learning_at_step=$BEGIN_LEARNING_AT_STEP --n_update_steps=$N_UPDATE_STEPS --buffer_type=normal --buffer_size=$BUFFER_SIZE --add_episode_time_state=True --prioritized_replay_compute_init=True --seed=9 --savedir=$SAVEDIR
python examples/cartpole_ddpg.py --num_steps=$NUMSTEPS --begin_learning_at_step=$BEGIN_LEARNING_AT_STEP --n_update_steps=$N_UPDATE_STEPS --buffer_type=normal --buffer_size=$BUFFER_SIZE --add_episode_time_state=True --prioritized_replay_compute_init=True --seed=10 --savedir=$SAVEDIR

