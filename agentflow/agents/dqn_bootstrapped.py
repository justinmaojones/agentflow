import tensorflow as tf

from .base_agent import BaseAgent
from ..tensorflow.ops import weighted_avg 
from ..tensorflow.ops import value_at_argmax 

class BootstrappedDQN(BaseAgent):

    def __init__(self,
            state_shape,
            num_actions,
            num_heads,
            q_fn,
            optimizer,
            q_prior_fn=None,
            double_q=False,
            random_prior=False,
            prior_scale=1.0,
        ):
        """Implements Boostrapped Deep Q Networks [1] with Tensorflow

        This class builds a DDPG model with optimization update and action prediction steps.

        Args:
          state_shape: a tuple or list of the state shape, excluding the batch dimension.
            For example, for images of size 28 x 28 x 3, state_shape=[28, 28, 3].
          num_actions: an integer, representing the number of possible actions an agent can choose
            from, excluding the batch dimension. It is assumed that actions are one-hot, 
            i.e. "one of `num_actions`".
          num_heads: an integer, representing the number of Q functions used for bootstrapping.
          q_fn: a function that takes as input two tensors: the state and action,
            and outputs an estimate Q(state, action)
          optimizer: tf.keras.optimizers.Optimizer.
          q_prior_fn: a function that takes as input two tensors: the state and action,
            and outputs an estimate Q(state, action)
          double_q: boolean, when true uses "double q-learning" from [2]. Otherwise uses
            standard q-learning.
          random_prior: boolean, when true builds a separate randomized prior function [3] 
            for each bootstrap head using q_fn.
          prior_scale: float, controls the strength of the prior
            
        References:
        [1] Osband, Ian, et al. "Deep exploration via bootstrapped DQN." Advances in neural 
            information processing systems. 2016.
        [2] Van Hasselt, Hado, Arthur Guez, and David Silver. "Deep reinforcement learning 
            with double q-learning." arXiv preprint arXiv:1509.06461 (2015).
        [3] Osband, Ian, John Aslanides, and Albin Cassirer. "Randomized prior functions for 
            deep reinforcement learning." Advances in Neural Information Processing Systems. 2018.
        """
        if q_prior_fn is not None:
            assert random_prior, "random_prior must be true if q_prior_fn is provided"

        self.state_shape = list(state_shape)
        self.num_actions = num_actions 
        self.num_heads = num_heads
        self.q_fn = q_fn
        self.optimizer = optimizer
        self.q_prior_fn = q_prior_fn if q_prior_fn is not None else q_fn
        self.double_q = double_q
        self.random_prior = random_prior
        self.prior_scale = prior_scale

        self.build_model()

    def _validate_q_fn(self, Q):
        if not (Q.shape.as_list() == [None, self.num_actions, self.num_heads]):
            err_msg = "expected q_fn to output tensor with shape [None, %d, %d], but instead got %s" % \
                        (self.num_actions, self.num_heads, str(Q.shape.as_list()))
            raise ValueError(err_msg)

    def build_net(self, name):
        state_input = tf.keras.Input(shape=tuple(self.state_shape))

        Q_multihead = self.q_fn(state_input, name=f"{name}/Q")
        self._validate_q_fn(Q_multihead)

        if self.random_prior:
            Q_prior_multihead = tf.stop_gradient(self.q_prior_fn(state_input, name=f"{name}/Q_prior"))
            self._validate_q_fn(Q_prior_multihead)
            Q_multihead = Q_multihead + self.prior_scale * Q_prior_multihead

        Q_model = tf.keras.Model(
            inputs=state_input,
            outputs=Q_multihead
        )

        return Q_model

    def build_model(self):

        with tf.name_scope('BootstrappedDQN'):

            inputs = {
                'state': tf.keras.Input(shape=tuple(self.state_shape), dtype=tf.float32, name='state'), 
                'action': tf.keras.Input(shape=(self.num_actions, ), dtype=tf.float32, name='action'), 
                'reward': tf.keras.Input(shape=(), dtype=tf.float32, name='reward'), 
                'done': tf.keras.Input(shape=(), dtype=tf.float32, name='done'), 
                'state2': tf.keras.Input(shape=tuple(self.state_shape), dtype=tf.float32, name='state2'), 
                'mask': tf.keras.Input(shape=(self.num_heads, ), dtype=tf.float32, name='mask')
            }
            self.inputs = inputs


            # dont use tf.keras.models.clone_model, because it will not preserve
            # uniqueness of shared objects within model
            Q_model = self.build_net("BoostrappedDQN/main")
            Q_model_target = self.build_net("BootstrappedDQN/target")

            self.weights = Q_model.weights
            self.trainable_weights = Q_model.trainable_weights
            self.non_trainable_weights = Q_model.non_trainable_weights

            self.weights_target = Q_model_target.weights 
            self.trainable_weights_target = Q_model_target.trainable_weights 
            self.non_trainable_weights_target = Q_model_target.non_trainable_weights 

            self.global_weights = self.weights + self.weights_target

            # the bootstrapped heads should be in the last dimension
            # Q_train_multihead is shape (None, num_actions, num_heads)
            Q_train_multihead = Q_model(inputs['state'], training=True)

            # Q_action_train_multihead is shape (None, num_actions, num_heads) 
            # and represents Q(s, a) for each head
            Q_action_train_multihead = tf.reduce_sum(
                    Q_train_multihead*inputs['action'][:, :, None], axis=-2)

            # Q_eval_multihead is shape (None, num_actions, num_heads)
            Q_eval_multihead = Q_model(inputs['state'], training=False)

            # Q_eval is shape (None, num_actions)
            Q_eval = tf.reduce_mean(Q_eval_multihead, axis=-1)
            Q_eval_masked = weighted_avg(Q_eval_multihead, inputs['mask'][:, None, :], axis=-1)
            Q_policy_eval = tf.reduce_max(Q_eval, axis=-1)

            # policy_eval is the index of the action with the highest Q(s,a)
            policy_eval = tf.argmax(Q_eval, axis=-1)
            policy_eval_masked = tf.argmax(Q_eval_masked, axis=-1)

            # Q_state2_eval_multihead is shape (None, num_actions, num_heads)
            Q_state2_eval_multihead = Q_model(inputs['state2'], training=False)
            # Q_state2_eval is shape (None, num_actions)
            Q_state2_eval = weighted_avg(Q_state2_eval_multihead, inputs['mask'][:, None, :])

            # Q_state2_target_multihead is shape (None, num_actions, num_heads)
            Q_state2_target_multihead = Q_model_target(inputs['state2'], training=False)

            # Q_state2_target_action is shape (None, num_heads)
            if self.double_q:
                # use action = argmax(Q_state2_eval_multihead, axis=-2) to select Q 
                # from axis -2 of Q_state2_target_multihead
                Q_state2_target_action = value_at_argmax(
                        Q_state2_eval_multihead, Q_state2_target_multihead, axis=-2) 
            else: 
                Q_state2_target_action = tf.reduce_max(Q_state2_target_multihead, axis=-2)

            # store attributes for later use
            self.train_outputs = {
                'policy_eval': policy_eval,
                'policy_eval_masked': policy_eval_masked,
                'Q_action_train_multihead': Q_action_train_multihead,
                'Q_eval': Q_eval,
                'Q_eval_masked': Q_eval_masked,
                'Q_policy_eval': Q_policy_eval,
                'Q_state2_eval': Q_state2_eval,
                'Q_state2_target_action': Q_state2_target_action,
            }

            self.policy_model = tf.keras.Model(inputs['state'], policy_eval)
            self.policy_masked_model = tf.keras.Model([inputs['state'], inputs['mask']], policy_eval_masked)
            self.policy_logits_model = tf.keras.Model(inputs['state'], Q_eval)
            self.policy_logits_masked_model = tf.keras.Model([inputs['state'], inputs['mask']], Q_eval_masked)
            self.train_model = tf.keras.Model(inputs, self.train_outputs)

    def _loss_fn(self, Q_action_train_multihead, reward, gamma, done, Q_state2_target_action, **kwargs):
        y = tf.stop_gradient(reward + gamma*done*Q_state2_target_action)
        td_error = Q_action_train_multihead - y
        loss = 0.5 * tf.square(td_error)
        return loss, (y, td_error)

    def compute_losses(self, reward, gamma, done, model_outputs):
        # loss functions
        losses_Q_multihead, (y, td_error_multihead) = self._loss_fn(
            model_outputs['Q_action_train_multihead'],
            reward[:, None],
            gamma,
            (1-done[:, None]),
            model_outputs['Q_state2_target_action'],
        )
        td_error = weighted_avg(td_error_multihead, mask) 

        # mask losses and normalize by number of active heads in mask
        # note: the original paper normalizes by total number of heads,
        # but this seemed more intuitive to me...choice probably doesn't
        # matter too much
        losses = weighted_avg(losses_Q_multihead, mask, axis=-1)
        
        addl_output = {
            'td_error': td_error,
            'y': y
        }
        return losses, addl_output

    @tf.function
    def policy_logits(self, state, mask=None):
        if mask is None:
            return self.policy_logits_model(state)
        else:
            return self.policy_logits_masked_model([state, mask])
        
