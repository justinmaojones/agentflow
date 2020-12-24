import numpy as np
import tensorflow as tf

def build_nested_placeholder(tf_type, shape):
    if isinstance(shape,dict) and isinstance(tf_type,dict):
        return {k: build_nested_placeholder(tf_type[k], shape[k])}
    elif isinstance(shape,dict):
        return {k: build_nested_placeholder(tf_type, shape[k])}
    else:
        return tf.placeholder(tf_type,shape=tuple([None]+shape))

def build_nested_feed_dict(inputs, placeholders):
    feed_dict = {}
    def func(inputs, placeholders):
        if isinstance(inputs,dict):
            if not isinstance(placeholders, dict):
                raise TypeError("inputs and placeholders nested structure should match")
            for k in inputs:
                func(inputs[k],placeholders[k])
        else:
            feed_dict[placeholders] = inputs
    func(inputs,placeholders)
    return feed_dict

def test_agent(test_env, agent):
    state = test_env.reset()
    rt = None
    all_done = 0
    while np.mean(all_done) < 1:
        action = agent.act(state)
        step_output = test_env.step(action)
        state = step_output['state']
        reward = step_output['reward']
        done = step_output['done']
        if rt is None:
            rt = reward.copy()
            all_done = done.copy()
        else:
            rt += reward*(1-all_done)
            all_done = np.maximum(done,all_done)
    return rt


