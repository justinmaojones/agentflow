import numpy as np
import tensorflow as tf


def test_agent(test_env, agent):
    state = test_env.reset()["state"]
    rt = None
    all_done = 0
    while np.mean(all_done) < 1:
        action = agent.act(state, explore=False).numpy()
        step_output = test_env.step(action)
        state = step_output["state"]
        reward = step_output["reward"]
        done = step_output["done"]
        if rt is None:
            rt = reward.copy()
            all_done = done.copy()
        else:
            rt += reward * (1 - all_done)
            all_done = np.maximum(done, all_done)
    return rt
