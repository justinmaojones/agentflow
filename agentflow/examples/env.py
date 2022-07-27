from agentflow.env import AtariGymEnv
from agentflow.env import ClippedRewardEnv
from agentflow.state import CropImageStateEnv
from agentflow.state import CvtRGB2GrayImageStateEnv
from agentflow.state import NPrevFramesStateEnv
from agentflow.state import ResizeImageStateEnv


def dqn_atari_paper_env(
    env_id, n_envs=4, n_prev_frames=4, frames_per_action=1, cropped=False
):
    env = AtariGymEnv(env_id, n_envs=n_envs, frames_per_action=frames_per_action)
    env = ClippedRewardEnv(env, -1, 1)
    env = CvtRGB2GrayImageStateEnv(env)
    if cropped:
        env = ResizeImageStateEnv(env, resized_shape=(84, 110))
        env = CropImageStateEnv(env, top=18, bottom=8)
    else:
        env = ResizeImageStateEnv(env, resized_shape=(84, 84))
    env = NPrevFramesStateEnv(env, n_prev_frames=n_prev_frames, flatten=True)
    return env
