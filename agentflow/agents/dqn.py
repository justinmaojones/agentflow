import numpy as np
import tensorflow as tf
from tensorflow.python.framework.errors_impl import InvalidArgumentError
from trfl import td_learning

from ..tensorflow.ops import l2_loss 
from ..tensorflow.ops import value_at_argmax 

class DQN(object):

    def __init__(self,
            state_shape,
            num_actions,
            q_fn,
            optimizer,
            double_q=False
        ):
        """Implements Deep Q Networks [1] with Tensorflow 

        This class builds a DDPG model with optimization update and action prediction steps.

        Args:
          state_shape: a tuple or list of the state shape, excluding the batch dimension.
            For example, for images of size 28 x 28 x 3, state_shape=[28, 28, 3].
          num_actions: an integer, representing the number of possible actions an agent can choose
            from, excluding the batch dimension. It is assumed that actions are one-hot, 
            i.e. "one of `num_actions`".
          q_fn: a function that takes as input two tensors: the state and action,
            and outputs an estimate Q(state, action)
          optimizer: tf.keras.optimizers.Optimizer.
          double_q: boolean, when true uses "double q-learning" from [2]. Otherwise uses
            standard q-learning.
            
        References:
        [1] Mnih, Volodymyr, et al. "Playing atari with deep reinforcement learning." 
            arXiv preprint arXiv:1312.5602 (2013).
        [2] Van Hasselt, Hado, Arthur Guez, and David Silver. "Deep reinforcement learning 
            with double q-learning." arXiv preprint arXiv:1509.06461 (2015).
        """
        self.state_shape = list(state_shape)
        self.num_actions = num_actions 
        self.q_fn = q_fn
        self.optimizer = optimizer
        self.double_q = double_q

        self.build_model()

    def build_net(self, name):
        state_input = tf.keras.Input(shape=tuple(self.state_shape))

        Q_model = tf.keras.Model(
            inputs=state_input,
            outputs=self.q_fn(state_input, name=f"{name}/Q")
        )

        return Q_model



    def build_model(self):

        with tf.name_scope('DQN'):

            inputs = {
                'state': tf.keras.Input(shape=tuple(self.state_shape), dtype=tf.float32, name='state'), 
                'action': tf.keras.Input(shape=(self.num_actions, ), dtype=tf.float32, name='action'), 
                'reward': tf.keras.Input(shape=(), dtype=tf.float32, name='reward'), 
                'done': tf.keras.Input(shape=(), dtype=tf.float32, name='done'), 
                'state2': tf.keras.Input(shape=tuple(self.state_shape), dtype=tf.float32, name='state2'), 
            }
            self.inputs = inputs

            # dont use tf.keras.models.clone_model, because it will not preserve
            # uniqueness of shared objects within model
            Q_model = self.build_net("DQN/main")
            Q_model_target = self.build_net("DQN/target")

            self.weights = Q_model.weights
            self.trainable_weights = Q_model.trainable_weights
            self.non_trainable_weights = Q_model.non_trainable_weights

            self.weights_target = Q_model_target.weights 
            self.trainable_weights_target = Q_model_target.trainable_weights 
            self.non_trainable_weights_target = Q_model_target.non_trainable_weights 


            Q_train = Q_model(inputs['state'], training=True)

            assert Q_train.shape.as_list() == inputs['action'].shape.as_list(), \
                "Q_train shape (%s) and action shape (%s) must match" % \
                (str(Q_train.shape), str(inputs['action'].shape))

            Q_action_train = tf.reduce_sum(Q_train*inputs['action'], axis=-1)

            Q_eval = Q_model(inputs['state'], training=False)

            if not (Q_eval.shape.as_list() == inputs['action'].shape.as_list()):
                raise InvalidArgumentError("Q_eval shape (%s) and action shape (%s) must match" % \
                                             (str(Q_eval.shape), str(inputs['action'].shape)))

            Q_action_eval = tf.reduce_sum(Q_train*inputs['action'], axis=-1, keepdims=True)
            policy_eval = tf.argmax(Q_eval, axis=-1)
            Q_policy_eval = tf.reduce_max(Q_eval, axis=-1)
            Q_state2_eval = Q_model(inputs['state2'], training=False)

            Q_state2_target = Q_model_target(inputs['state2'], training=False)

            if self.double_q:
                Q_state2_target_action = value_at_argmax(Q_state2_eval, Q_state2_target, axis=-1)
            else:
                Q_state2_target_action = tf.reduce_max(Q_state2_target, axis=-1)

            # store attributes for later use
            self.train_outputs = {
                'policy_eval': policy_eval,
                'Q_action_eval': Q_action_eval,
                'Q_action_train': Q_action_train,
                'Q_policy_eval': Q_policy_eval,
                'Q_state2_eval': Q_state2_eval,
                'Q_state2_target': Q_state2_target,
                'Q_state2_target_action': Q_state2_target_action,
            }

            self.policy_model = tf.keras.Model(inputs['state'], policy_eval)
            self.policy_logits_model = tf.keras.Model(inputs['state'], Q_eval)
            self.train_model = tf.keras.Model(inputs, self.train_outputs)

    def l2_loss(self, weight_decay):
        return 0.5 * tf.reduce_sum([tf.nn.l2_loss(x) for x in self.trainable_weights])

    @tf.function
    def act(self, state):
        return self.policy_model(state)

    @tf.function
    def policy_logits(self, state):
        return self.policy_logits_model(state)
        
    @tf.function
    def update(self, 
            state, 
            action, 
            reward, 
            done, 
            state2, 
            gamma=0.99, 
            ema_decay=0.999, 
            weight_decay=None, 
            grad_clip_norm=None,
            importance_weight=None, 
            outputs=['td_error'], 
        ):

        # autocast types
        reward = tf.cast(reward, tf.float32)
        done = tf.cast(done, tf.float32)
        gamma = tf.cast(gamma, tf.float32)
        ema_decay = tf.cast(ema_decay, tf.float32)
        if weight_decay is not None:
            weight_decay = tf.cast(weight_decay, tf.float32)
        if importance_weight is not None:
            importance_weight = tf.cast(importance_weight, tf.float32)
        if grad_clip_norm is not None:
            grad_clip_norm = tf.cast(grad_clip_norm, tf.float32)

        with tf.GradientTape() as tape:
            # do not watch target network weights
            tape.watch(self.trainable_weights)

            model_outputs = self.train_model({
                'state': state,
                'action': action,
                'reward': reward,
                'done': done,
                'state2': state2,
            })

            # loss functions
            losses, (y, td_error) = td_learning(
                model_outputs['Q_action_train'],
                reward,
                gamma*(1-done),
                model_outputs['Q_state2_target_action'],
            )

            # check shapes
            tf.debugging.assert_rank(losses, 1)

            if importance_weight is None:
                loss = tf.reduce_mean(losses)
            else:
                # check shapes
                tf.debugging.assert_equal(
                    tf.shape(importance_weight),
                    tf.shape(losses),
                    message = "shape of importance_weight and losses do not match")

                # overall loss function (importance weighted)
                loss = tf.reduce_mean(importance_weight * losses)

            if weight_decay is not None:
                loss = loss + weight_decay * l2_loss(self.trainable_weights)

        # update model weights
        grads = tape.gradient(loss, self.trainable_weights)

        # gradient clipping
        if grad_clip_norm is not None:
            grads, gnorm = tf.clip_by_global_norm(grads, grad_clip_norm)
        else:
            gnorm = tf.linalg.global_norm(grads)

        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        # ema update, zero debias not needed since we initialized identically
        for v, vt in zip(self.trainable_weights, self.trainable_weights_target):
            vt.assign(ema_decay*vt + (1.0-ema_decay)*v)

        # e.g. batchnorm moving avg should be same between main & target models
        for v, vt in zip(self.non_trainable_weights, self.non_trainable_weights_target):
            vt.assign(v)

        # parameter norms
        pnorm = tf.linalg.global_norm(self.trainable_weights)
        pnorm_target = tf.linalg.global_norm(self.trainable_weights_target)
        pnorm_nt = tf.linalg.global_norm(self.non_trainable_weights)
        pnorm_target_nt = tf.linalg.global_norm(self.non_trainable_weights_target)

        model_outputs['pnorm/main/trainable_weights'] = pnorm
        model_outputs['pnorm/target/trainable_weights'] = pnorm_target
        model_outputs['pnorm/main/non_trainable_weights'] = pnorm_nt
        model_outputs['pnorm/target/non_trainable_weights'] = pnorm_target_nt
        model_outputs['gnorm'] = gnorm
        model_outputs['losses'] = losses
        model_outputs['td_error'] = td_error
        model_outputs['y'] = y

        # collect output
        rvals = {}
        rvals['loss'] = loss
        for k in outputs:
            rvals[k] = model_outputs[k]

        return rvals
