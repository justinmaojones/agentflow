from abc import abstractmethod
from dataclasses import dataclass
import tensorflow as tf
from typing import Dict, List

from agentflow.agents.source import AgentSource
from ..tensorflow.ops import l2_loss 


@dataclass
class BaseAgent(AgentSource):

    policy_model: tf.keras.Model
    train_model: tf.keras.Model

    optimizer: tf.keras.optimizers.Optimizer

    weights: List[tf.Variable]
    trainable_weights: List[tf.Variable]
    non_trainable_weights: List[tf.Variable]

    weights_target: List[tf.Variable]
    trainable_weights_target: List[tf.Variable]
    non_trainable_weights_target: List[tf.Variable]

    def __hash__(self):
        """classes become unhashable with keras model attributes"""
        return hash(self.__class__.__name__)

    @tf.function
    def act(self, state, mask=None):
        if mask is None:
            return self.policy_model(state)
        else:
            return self.policy_model([state, mask])
        
    @abstractmethod
    def build_model(self):
        ...

    @abstractmethod
    def compute_losses(self, model_outputs, reward, gamma, done, mask=None):
        ...

    def get_weights(self):
        return self.train_model.get_weights()

    @property
    def global_weights(self) -> List[tf.Variable]:
        return self.weights + self.weights_target

    @property
    def learning_rate(self):
        if isinstance(self.optimizer.learning_rate, tf.Variable):
            return self.optimizer.learning_rate
        elif isinstance(self.optimizer.learning_rate, tf.keras.optimizers.schedules.LearningRateSchedule):
            return self.optimizer.learning_rate(self.optimizer.iterations)
        else:
            raise NotImplementedError("Unhandled type of learning rate in {self}")

    def load_weights(self, filepath):
        self.train_model.load_weights(filepath)

    @tf.function
    def pnorms(self, per_layer=False):
        """
        Returns the 2-norm of every trainable and non trainable weight
        in main and target graphs. Typically useful for debugging.
        """

        weight_classes = {
            'trainable_weights/main': self.trainable_weights,
            'trainable_weights/target': self.trainable_weights_target,
            'non_trainable_weights/main': self.non_trainable_weights,
            'non_trainable_weights/target': self.non_trainable_weights_target,
        }

        pnorms = {}
        if per_layer:
            for c in weight_classes:
                weights = weight_classes[c]
                for k in weights:
                    w = self.trainable_weights[k]
                    pnorms[f"pnorms/{c}/{w.name}"] = tf.norm(w)

        # overall parameter norms
        for c in weight_classes:
            pnorms[f"pnorm/overall/{c}"] = tf.linalg.global_norm(weight_classes[c])

        return pnorms

    def save_weights(self, filepath):
        self.train_model.save_weights(filepath)

    def set_weights(self, weights):
        self.train_model.set_weights(weights)

    @tf.function
    def update(self,
            state,
            action,
            reward,
            done,
            state2,
            mask=None,
            gamma=0.99,
            ema_decay=0.999,
            weight_decay=None,
            grad_clip_norm=None,
            importance_weight=None,
            debug=False
        ):

        outputs = {}
        def add_to_outputs(x: Dict):
            for k in x:
                assert k not in outputs, f"{k} is already in outputs"
                outputs[k] = x[k]

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

            model_inputs = {
                'state': state,
                'action': action,
                'reward': reward,
                'done': done,
                'state2': state2,
            }

            if mask is not None:
                mask = tf.cast(mask, tf.float32)
                model_inputs['mask'] = mask

            model_outputs = self.train_model(model_inputs)
            add_to_outputs(model_outputs)

            losses, addl_losses_output = self.compute_losses(model_outputs, reward, gamma, done, mask)
            add_to_outputs({'losses': losses})
            add_to_outputs(addl_losses_output)

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

            add_to_outputs({'loss': loss})

        # update model weights
        grads = tape.gradient(loss, self.trainable_weights)

        # gradient clipping
        if grad_clip_norm is not None:
            grads, gnorm = tf.clip_by_global_norm(grads, grad_clip_norm)
        else:
            gnorm = tf.linalg.global_norm(grads)

        add_to_outputs({'gnorm': gnorm})

        add_to_outputs({'learning_rate': self.learning_rate})
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        # ema update, zero debias not needed since we initialized identically
        for v, vt in zip(self.trainable_weights, self.trainable_weights_target):
            vt.assign(ema_decay*vt + (1.0-ema_decay)*v)

        # e.g. batchnorm moving avg should be same between main & target models
        for v, vt in zip(self.non_trainable_weights, self.non_trainable_weights_target):
            vt.assign(v)

        if debug:
            return {f"update/{k}": outputs[k] for k in outputs}
        else:
            keys = ['loss', 'learning_rate']
            return {f"update/{k}": outputs[k] for k in keys}


