import click
import h5py
import numpy as np
import os
import tensorflow as tf
import time
import yaml

from agentflow.env import CartpoleGymEnv
from agentflow.agents import BootstrappedDQN
from agentflow.agents import CompletelyRandomDiscreteUntil
from agentflow.agents import EpsilonGreedy
from agentflow.buffers import BootstrapMaskBuffer
from agentflow.buffers import BufferMap
from agentflow.buffers import PrioritizedBufferMap
from agentflow.buffers import NStepReturnBuffer
from agentflow.logging import scoped_log_tf_summary
from agentflow.numpy.schedules import ExponentialDecaySchedule
from agentflow.numpy.schedules import LinearAnnealingSchedule
from agentflow.state import NPrevFramesStateEnv
from agentflow.env import PrevEpisodeReturnsEnv
from agentflow.env import PrevEpisodeLengthsEnv
from agentflow.state import RandomOneHotMaskEnv
from agentflow.tensorflow.nn import dense_net
from agentflow.tensorflow.nn import normalize_ema
from agentflow.train import Trainer


@click.option("--num_steps", default=30000, type=int)
@click.option("--n_envs", default=1)
@click.option("--n_prev_frames", default=16)
@click.option("--ema_decay", default=0.99, type=float)
@click.option(
    "--noise", default="eps_greedy", type=click.Choice(["eps_greedy", "gumbel_softmax"])
)
@click.option("--noise_eps", default=0.05, type=float)
@click.option("--noise_temperature", default=1.0, type=float)
@click.option("--double_q", default=True, type=bool)
@click.option("--bootstrap_num_heads", default=16, type=int)
@click.option("--bootstrap_mask_prob", default=0.5, type=float)
@click.option("--bootstrap_prior_scale", default=1.0, type=float)
@click.option("--bootstrap_random_prior", default=False, type=bool)
@click.option("--hidden_dims", default=64, type=int)
@click.option("--hidden_layers", default=4, type=int)
@click.option("--normalize_inputs", default=True, type=bool)
@click.option("--batchnorm", default=False, type=bool)
@click.option(
    "--buffer_type",
    default="normal",
    type=click.Choice(["normal", "prioritized", "delayed", "delayed_prioritized"]),
)
@click.option("--buffer_size", default=30000, type=int)
@click.option("--enable_n_step_return_publisher", default=True, type=bool)
@click.option("--n_step_return", default=8, type=int)
@click.option("--prioritized_replay_alpha", default=0.6, type=float)
@click.option("--prioritized_replay_beta0", default=0.4, type=float)
@click.option("--prioritized_replay_beta_iters", default=None, type=int)
@click.option("--prioritized_replay_eps", default=1e-6, type=float)
@click.option("--prioritized_replay_simple", default=True, type=bool)
@click.option("--prioritized_replay_default_reward_priority", default=5, type=float)
@click.option("--prioritized_replay_default_done_priority", default=5, type=float)
@click.option("--begin_learning_at_step", default=200)
@click.option("--learning_rate", default=1e-4)
@click.option("--learning_rate_decay", default=0.99995)
@click.option("--adam_eps", default=1e-7)
@click.option("--gamma", default=0.99)
@click.option("--weight_decay", default=1e-4)
@click.option("--entropy_loss_weight", default=1e-5, type=float)
@click.option("--entropy_loss_weight_decay", default=0.99995)
@click.option("--n_update_steps", default=4, type=int)
@click.option("--update_freq", default=1, type=int)
@click.option("--n_steps_per_eval", default=100, type=int)
@click.option("--batchsize", default=64)
@click.option("--savedir", default="results")
@click.option("--seed", default=None, type=int)
def run(**cfg):

    for k in sorted(cfg):
        print("CONFIG: ", k, str(cfg[k]))

    if cfg["seed"] is not None:
        np.random.seed(cfg["seed"])
        tf.random.set_seed(int(10 * cfg["seed"]))

    ts = str(int(time.time() * 1000))
    suff = str(np.random.choice(np.iinfo(np.int64).max))
    savedir = os.path.join(cfg["savedir"], "experiment" + ts + "_" + suff)
    print("SAVING TO: {savedir}".format(**locals()))
    os.system("mkdir -p {savedir}".format(**locals()))

    with open(os.path.join(savedir, "config.yaml"), "w") as f:
        yaml.dump(cfg, f)

    log = scoped_log_tf_summary(savedir)

    # environment
    env = CartpoleGymEnv(n_envs=cfg["n_envs"])
    env = NPrevFramesStateEnv(env, n_prev_frames=cfg["n_prev_frames"], flatten=True)
    env = RandomOneHotMaskEnv(env, depth=cfg["bootstrap_num_heads"])
    env = PrevEpisodeReturnsEnv(env)
    env = PrevEpisodeLengthsEnv(env)
    test_env = CartpoleGymEnv(n_envs=1)
    test_env = NPrevFramesStateEnv(
        test_env, n_prev_frames=cfg["n_prev_frames"], flatten=True
    )

    # state and action shapes
    state = env.reset()["state"]
    state_shape = state.shape
    print("STATE SHAPE: ", state_shape)

    num_actions = 2
    print("ACTION SHAPE: ", num_actions)

    # build agent
    def q_fn(state, name=None, **kwargs):
        if cfg["normalize_inputs"]:
            state = normalize_ema(state, name=name)
        h = dense_net(
            state,
            cfg["hidden_dims"],
            cfg["hidden_layers"],
            batchnorm=cfg["batchnorm"],
            name=name,
        )
        output = tf.keras.layers.Dense(
            num_actions * cfg["bootstrap_num_heads"],
            name=f"{name}/dense/output" if name is not None else "dense/output",
        )(h)
        return tf.reshape(output, [-1, num_actions, cfg["bootstrap_num_heads"]])

    learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=cfg["learning_rate"],
        decay_rate=cfg["learning_rate_decay"],
        decay_steps=cfg["n_update_steps"] / cfg["update_freq"],
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate_schedule,
        epsilon=cfg["adam_eps"],
    )

    agent = BootstrappedDQN(
        state_shape=state_shape[1:],
        num_actions=num_actions,
        q_fn=q_fn,
        optimizer=optimizer,
        double_q=cfg["double_q"],
        num_heads=cfg["bootstrap_num_heads"],
        random_prior=cfg["bootstrap_random_prior"],
        prior_scale=cfg["bootstrap_prior_scale"],
        gamma=cfg["gamma"],
        ema_decay=cfg["ema_decay"],
        weight_decay=cfg["weight_decay"],
    )
    test_agent = agent

    if cfg["noise"] == "eps_greedy":
        agent = EpsilonGreedy(agent, epsilon=cfg["noise_eps"])
    else:
        raise NotImplementedError
    agent = CompletelyRandomDiscreteUntil(
        agent, num_steps=cfg["begin_learning_at_step"]
    )

    # Replay Buffer
    if cfg["buffer_type"] == "prioritized":
        # prioritized experience replay
        replay_buffer = PrioritizedBufferMap(
            max_length=cfg["buffer_size"],
            alpha=cfg["prioritized_replay_alpha"],
            eps=cfg["prioritized_replay_eps"],
            default_priority=1.0,
            default_non_zero_reward_priority=cfg[
                "prioritized_replay_default_reward_priority"
            ],
            default_done_priority=cfg["prioritized_replay_default_done_priority"],
        )

        beta_schedule = LinearAnnealingSchedule(
            initial_value=cfg["prioritized_replay_beta0"],
            final_value=1.0,
            annealing_steps=cfg["prioritized_replay_beta_iters"] or cfg["num_steps"],
            begin_at_step=cfg["begin_learning_at_step"],
        )

    else:
        # Normal Buffer
        replay_buffer = BufferMap(cfg["buffer_size"])

    # Delays publishing of records to the underlying replay buffer for n steps
    # then publishes the discounted n-step return
    if cfg["enable_n_step_return_publisher"]:
        replay_buffer = NStepReturnBuffer(
            replay_buffer,
            n_steps=cfg["n_step_return"],
            gamma=cfg["gamma"],
        )

    replay_buffer = BootstrapMaskBuffer(
        replay_buffer,
        depth=cfg["bootstrap_num_heads"],
        sample_prob=cfg["bootstrap_mask_prob"],
    )

    trainer = Trainer(
        env=env,
        agent=agent,
        replay_buffer=replay_buffer,
        batchsize=cfg["batchsize"],
        test_env=test_env,
        test_agent=test_agent,
        log=log,
        begin_learning_at_step=cfg["begin_learning_at_step"],
        n_steps_per_eval=cfg["n_steps_per_eval"],
        update_freq=cfg["update_freq"],
        n_update_steps=cfg["n_update_steps"],
    )
    trainer.learn(cfg["num_steps"])

    log.flush()


if __name__ == "__main__":
    click.command()(run)()
