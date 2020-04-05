# Copied from https://github.com/deepmind/trfl/blob/master/trfl/value_ops.py
#
# Copyright 2018 The trfl Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""TensorFlow ops for state value learning."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

# Dependency imports
import tensorflow as tf


def td_learning_non_episodic(v_tm1, r_t, r_avg, v_t, name="TDLearning"):
  """Implements the TD(0)-learning loss as a TensorFlow op.
  The TD loss is `0.5` times the squared difference between `v_tm1` and
  the target `r_t + pcont_t * v_t`.
  See "Learning to Predict by the Methods of Temporal Differences" by Sutton.
  (https://link.springer.com/article/10.1023/A:1022633531479).
  Args:
    v_tm1: Tensor holding values at previous timestep, shape `[B]`.
    r_t: Tensor holding rewards, shape `[B]`.
    pcont_t: Tensor holding pcontinue values, shape `[B]`.
    v_t: Tensor holding values at current timestep, shape `[B]`.
    name: name to prefix ops created by this function.
  Returns:
    A namedtuple with fields:
    * `loss`: a tensor containing the batch of losses, shape `[B]`.
    * `extra`: a namedtuple with fields:
        * `target`: batch of target values for `v_tm1`, shape `[B]`.
        * `td_error`: batch of temporal difference errors, shape `[B]`.
  """

  # TD(0)-learning op.
  with tf.name_scope(name, values=[v_tm1, r_avg, r_t, v_t]):

    # Build target.
    target_v = tf.stop_gradient(r_t - r_avg + v_t)
    target_r = tf.stop_gradient(r_t)

    # Temporal difference error and loss.
    # Loss is MSE scaled by 0.5, so the gradient is equal to the TD error.
    td_error_v = target_v - v_tm1
    error_r = target_r - r_avg
    loss_v = 0.5 * tf.square(td_error_v)
    loss_r = 0.5 * tf.square(error_r)
    loss = loss_v + loss_r
    return loss, target_v, target_r, td_error_v, error_r