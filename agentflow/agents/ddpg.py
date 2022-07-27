import tensorflow as tf
from trfl import td_learning
from trfl import dpg

from .base_agent import BaseAgent
from ..tensorflow.ops import l2_loss


class DDPG(BaseAgent):
    def __init__(
        self,
        state_shape,
        action_shape,
        policy_fn,
        q_fn,
        optimizer,
        dqda_clipping=None,
        clip_norm=False,
        policy_loss_weight=1.0,
        auto_build: bool = True,
        **kwargs,
    ):
        """Implements Deep Deterministic Policy Gradient with Tensorflow

        This class builds a DDPG model with optimization update and action prediction steps.

        Args:
          state_shape: a tuple or list of the state shape, excluding the batch dimension.
            For example, for images of size 28 x 28 x 3, state_shape=[28,28,3].
          action_shape: a tuple or list of the action shape, excluding the batch dimension.
            For example, for scalar actions, action_shape=[].  For a vector of actions
            with 3 elements, action_shape[3].
          policy_fn: a function that takes as input a tensor, the state, and
            outputs an action (with shape=action_shape, excluding batch dimension).
          q_fn: a function that takes as input two tensors: the state and action,
            and outputs an estimate Q(state,action)
          optimizer: tf.keras.optimizers.Optimizer.
          dqda_clipping: `int` or `float`, clips the gradient dqda element-wise
            between `[-dqda_clipping, dqda_clipping]`.
          clip_norm: Whether to perform dqda clipping on the vector norm of the last
            dimension, or component wise (default).
          policy_loss_weight: coefficient applied to policy loss

        References:
        [1] Barth-Maron, Gabriel, et al. "Distributed distributional deterministic
            policy gradients." arXiv preprint arXiv:1804.08617 (2018).
        """
        super().__init__(**kwargs)
        self.state_shape = list(state_shape)
        self.action_shape = list(action_shape)
        self.policy_fn = policy_fn
        self.q_fn = q_fn
        self.optimizer = optimizer
        self.dqda_clipping = dqda_clipping
        self.clip_norm = clip_norm
        self.policy_loss_weight = policy_loss_weight

        if auto_build:
            self.build_model()

    def build_net(self, name):
        state_input = tf.keras.Input(shape=tuple(self.state_shape))
        action_input = tf.keras.Input(shape=tuple(self.action_shape))

        policy_model = tf.keras.Model(
            inputs=state_input,
            outputs=self.policy_fn(state_input, name=f"{name}/policy"),
        )

        Q_model = tf.keras.Model(
            inputs=[state_input, action_input],
            outputs=self.q_fn(state_input, action_input, name=f"{name}/Q"),
        )

        return policy_model, Q_model

    def build_model(self):

        with tf.name_scope("DDPG"):

            # inputs
            inputs = {
                "state": tf.keras.Input(
                    shape=tuple(self.state_shape), dtype=tf.float32, name="state"
                ),
                "action": tf.keras.Input(
                    shape=tuple(self.action_shape), dtype=tf.float32, name="action"
                ),
                "reward": tf.keras.Input(shape=(), dtype=tf.float32, name="reward"),
                "done": tf.keras.Input(shape=(), dtype=tf.float32, name="done"),
                "state2": tf.keras.Input(
                    shape=tuple(self.state_shape), dtype=tf.float32, name="state2"
                ),
            }
            self.inputs = inputs

            # dont use tf.keras.models.clone_model, because it will not preserve
            # uniqueness of shared objects within model
            policy_model, Q_model = self.build_net("main")
            policy_model_target, Q_model_target = self.build_net("target")

            self.weights = policy_model.weights + Q_model.weights
            self.trainable_weights = (
                policy_model.trainable_weights + Q_model.trainable_weights
            )
            self.non_trainable_weights = (
                policy_model.non_trainable_weights + Q_model.non_trainable_weights
            )

            self.weights_target = policy_model_target.weights + Q_model_target.weights
            self.trainable_weights_target = (
                policy_model_target.trainable_weights + Q_model_target.trainable_weights
            )
            self.non_trainable_weights_target = (
                policy_model_target.non_trainable_weights + Q_model_target.non_trainable_weights
            )

            # initialize target weights equal to main model
            for v, vt in zip(self.trainable_weights, self.trainable_weights_target):
                vt.assign(v)

            policy_train = policy_model(inputs["state"], training=True)
            policy_eval = policy_model(inputs["state"], training=False)

            # Q_policy_train has training=False, because it it's purpose is for training
            # the policy model, not the Q model
            Q_policy_train = Q_model([inputs["state"], policy_train], training=False)
            Q_policy_eval = Q_model([inputs["state"], policy_eval], training=False)

            Q_action_train = Q_model([inputs["state"], inputs["action"]], training=True)

            policy_target = policy_model_target(inputs["state"], training=False)
            policy_target_state2 = policy_model_target(inputs["state2"], training=False)

            Q_target_state2 = Q_model_target(
                [inputs["state2"], policy_target_state2], training=False
            )

            # make sure inputs to loss functions are in the correct shape
            # (to avoid erroneous broadcasting)
            Q_action_train = tf.reshape(Q_action_train, [-1])
            Q_target_state2 = tf.reshape(Q_target_state2, [-1])

            # store attributes for later use
            self.train_outputs = {
                "policy_target": policy_target,
                "policy_target_state2": policy_target_state2,
                "policy_eval": policy_eval,
                "policy_train": policy_train,
                "Q_action_train": Q_action_train,
                "Q_target_state2": Q_target_state2,
                "Q_policy_eval": Q_policy_eval,
                "Q_policy_train": Q_policy_train,
            }

            self.policy_model = tf.keras.Model(inputs["state"], policy_eval)
            self.train_model = tf.keras.Model(inputs, self.train_outputs)

    def compute_losses(self, model_outputs, reward, gamma, done, mask):
        # loss functions
        losses_Q, (y, td_error) = td_learning(
            model_outputs["Q_action_train"],
            reward,
            gamma * (1 - done),
            model_outputs["Q_target_state2"],
        )
        losses_policy, _ = dpg(
            model_outputs["Q_policy_train"],
            model_outputs["policy_train"],
            self.dqda_clipping,
            self.clip_norm,
        )

        # check shapes
        tf.debugging.assert_rank(losses_Q, 1)
        tf.debugging.assert_rank(losses_policy, 1)

        # per sample overall loss function
        losses = losses_Q + self.policy_loss_weight * losses_policy

        addl_output = {"td_error": td_error, "y": y}
        return losses, addl_output
