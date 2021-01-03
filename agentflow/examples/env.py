from agentflow.env import VecGymEnv
from agentflow.state import CropImageStateEnv
from agentflow.state import CvtRGB2GrayImageStateEnv
from agentflow.state import NPrevFramesStateEnv
from agentflow.state import PrevEpisodeReturnsEnv 
from agentflow.state import PrevEpisodeLengthsEnv 
from agentflow.state import ResizeImageStateEnv

def dqn_atari_paper_env(env_id, n_envs=4, noops=30, fire_reset=False, n_prev_frames=4):
    env = VecGymEnv(env_id, n_envs=n_envs, noops=noops, fire_reset=fire_reset)
    env = PrevEpisodeReturnsEnv(env)
    env = PrevEpisodeLengthsEnv(env)
    env = CvtRGB2GrayImageStateEnv(env)
    env = ResizeImageStateEnv(env,resized_shape=(84,110))
    env = CropImageStateEnv(env,top=18,bottom=8)
    env = NPrevFramesStateEnv(env,n_prev_frames=n_prev_frames,flatten=True)
    return env
