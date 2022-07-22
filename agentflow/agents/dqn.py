import numpy as np
import tensorflow as tf
from tensorflow.python.framework.errors_impl import InvalidArgumentError
from trfl import td_learning

from .base_agent import BaseAgent
from .source import DiscreteActionAgentSource
from ..tensorflow.ops import l2_loss 
from ..tensorflow.ops import value_at_argmax 

class DQN(BaseAgent, DiscreteActionAgentSource):

    def __init__(self,
            state_shape,
            num_actions,
            q_fn,
            optimizer,
            double_q: bool = False,
            loss_type: str = 'huber',
            auto_build: bool = True,
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
          auto_build: boolean, automatically build model on class instantiation
            
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
        self.loss_type = loss_type
        self.auto_build = auto_build

        if auto_build:
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

            #Q_action_eval = tf.reduce_sum(Q_train*inputs['action'], axis=-1, keepdims=True)
            policy_eval = tf.argmax(Q_eval, axis=-1)
            #Q_policy_eval = tf.reduce_max(Q_eval, axis=-1)

            Q_state2_target = Q_model_target(inputs['state2'], training=False)

            if self.double_q:
                Q_state2_eval = Q_model(inputs['state2'], training=False)
                Q_state2_target_action = value_at_argmax(Q_state2_eval, Q_state2_target, axis=-1)
            else:
                Q_state2_target_action = tf.reduce_max(Q_state2_target, axis=-1)

            # store attributes for later use
            self.train_outputs = {
                    #'policy_eval': policy_eval,
                    #'Q_action_eval': Q_action_eval,
                'Q_action_train': Q_action_train,
                #'Q_policy_eval': Q_policy_eval,
                #'Q_state2_eval': Q_state2_eval,
                #'Q_state2_target': Q_state2_target,
                'Q_state2_target_action': Q_state2_target_action,
            }

            self.policy_model = tf.keras.Model(inputs['state'], policy_eval)
            self.policy_logits_model = tf.keras.Model(inputs['state'], Q_eval)
            self.train_model = tf.keras.Model(inputs, self.train_outputs)

    def huber_loss(self, y_true, y_pred, delta=1.):
        error = tf.subtract(y_pred, y_true)
        abs_error = tf.abs(error)
        half = tf.convert_to_tensor(0.5, dtype=abs_error.dtype)
        return tf.where(abs_error <= delta, half * tf.square(error),
                             delta * abs_error - half * tf.square(delta))

    def td_huber_loss(self, model_outputs, reward, gamma, done, mask):
        y = tf.stop_gradient(reward + gamma*(1-done) * model_outputs['Q_state2_target_action'])
        pred = model_outputs['Q_action_train']
        td_error = y - pred
        losses = 0.5 * self.huber_loss(y, pred)

        addl_output = {
            'td_error': td_error,
            'y': y
        }
        return losses, addl_output

    def td_mse_loss(self, model_outputs, reward, gamma, done, mask):
        # loss functions
        losses, (y, td_error) = td_learning(
            model_outputs['Q_action_train'],
            reward,
            gamma*(1-done),
            model_outputs['Q_state2_target_action'],
        )

        addl_output = {
            'td_error': td_error,
            'y': y
        }
        return losses, addl_output

    def compute_losses(self, model_outputs, reward, gamma, done, mask):

        if self.loss_type == 'mse':
            return self.td_mse_loss(model_outputs, reward, gamma, done, mask)

        elif self.loss_type == 'huber':
            return self.td_huber_loss(model_outputs, reward, gamma, done, mask)

        else:
            raise NotImplementedError(f"unhandled loss type {self.loss_type}")
