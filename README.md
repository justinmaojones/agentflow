![Build](https://github.com/justinmaojones/agentflow/workflows/Build/badge.svg)
![License](https://img.shields.io/badge/license-MIT-green)

# AgentFlow

An RL framework inspired by the composability of TensorFlow.  Primarily written for fun and learning.

## Installation

```
git clone git@github.com:justinmaojones/agentflow.git
cd agentflow
pip install .
```

## Quickstart

Construct an environment flow
```python
from agentflow.env import CartpoleGymEnv
from agentflow.state import NPrevFramesStateEnv

env = CartpoleGymEnv(n_envs=16)
env = NPrevFramesStateEnv(env, n_prev_frames=16, flatten=True) # appends prev frames to state

test_env = CartpoleGymEnv(n_envs=1)
test_env = NPrevFramesStateEnv(test_env, n_prev_frames=16, flatten=True) # appends prev frames to state
```

Construct a buffer flow
```python
from agentflow.buffers import BufferMap
from agentflow.buffers import NStepReturnBuffer

replay_buffer = BufferMap(max_length=30000)
replay_buffer = NStepReturnBuffer(replay_buffer, n_steps=8, gamma=0.99) # n-step discounted sum of rewards
```

Construct an agent flow
```python
import tensorflow as tf
from agentflow.agents import DQN
from agentflow.agents import CompletelyRandomDiscreteUntil
from agentflow.agents import EpsilonGreedy
from agentflow.tensorflow.nn import dense_net
from agentflow.tensorflow.nn import normalize_ema

def q_fn(state, **kwargs):
    state = normalize_ema(state)
    h = dense_net(state, units=64, layers=4)
    return tf.keras.layers.Dense(2)(h)

optimizer = tf.keras.optimizers.Adam(1e-4)
agent = DQN(state_shape=[64], num_actions=2, q_fn=q_fn, optimizer=optimizer, gamma=0.99, target_update_freq=16)
test_agent = agent
agent = EpsilonGreedy(agent, epsilon=0.5) # cartpole likes a lot of noise
agent = CompletelyRandomDiscreteUntil(agent, num_steps=1000) # uniform random actions until num_steps
```

Train using a Trainer
```python
from agentflow.train import Trainer
from agentflow.logging import scoped_log_tf_summary

log = scoped_log_tf_summary("/tmp/agentflow/cartpole")
trainer = Trainer(
    env, agent, replay_buffer, 
    log=log, batchsize=64, begin_learning_at_step=1000, 
    test_env=test_env, test_agent=test_agent)
trainer.learn(num_steps=10000)
```

or an AsyncTrainer 
```python
from agentflow.train import AsyncTrainer

async_trainer = AsyncTrainer(env, agent, replay_buffer, log=log, batchsize=64, begin_learning_at_step=1000, n_updates_per_model_refresh=32)
async_trainer.learn(num_updates=10000)

# interrupt async trainer with `ray stop` on commandline
```

View your results on Tensorboard. The trainers include tracing with [TensorFlow profilers](https://www.tensorflow.org/tensorboard/tensorboard_profiling_keras) to help diagnose performance bottlenecks.  To see results, go to the `Profile` tab in Tensorboard.
```
tensorboard --logdir /tmp/agentflow/cartpole
```

## More Examples

Check out [agentflow/examples](https://github.com/justinmaojones/agentflow/tree/master/agentflow/examples) and corresponding run scripts at [scripts/examples](https://github.com/justinmaojones/agentflow/tree/master/scripts/examples).

## Supported Features

There are many features available, here is a list of some of them
* Enviroments
    * [OpenAI Gym](https://github.com/openai/gym) (including vectorized gym environments)
    * Simple concave function environments (great for testing)
    * Chain game (from the [Boostrapped DQN paper](https://papers.nips.cc/paper/2016/file/8d8818c8e140c64c743113f563cf750f-Paper.pdf))
    * ...many more env flow wrappers...
* Replay Buffers
    * Prioritized experience replay buffers, using [STArr](https://github.com/justinmaojones/starr) for efficient sampling
    * Image compression buffers
* Agents supported
    * DQN (with double Q learning)
    * Bootstrapped DQN (with random prior functions)
    * DDPG
* Training
    * Synchronous Trainer
    * Asynchronous Trainer using the excellent [Ray](https://github.com/ray-project/ray) library 
* Logging to h5 as well as tensorboard

## To-Do
* Add tf.data pipeline to `Trainer` for performance optimization
* A recent refactor removed ability to update priorities for Prioritized Experience Replay...need to fix that
* Migrate chain examples to trainre and cleanup chain env
* Add more explicitly defined target updaters decoupled from agent classes
* Configuration is a bit messy, should clean that up
* Publish to pypi
