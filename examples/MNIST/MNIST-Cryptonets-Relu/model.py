# ******************************************************************************
# Copyright 2018-2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# *****************************************************************************

import tensorflow as tf
import numpy as np
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mnist_util import load_mnist_data, \
    get_variable, \
    conv2d_stride_2_valid, \
    avg_pool_3x3_same_size


def cryptonets_relu_model(x, mode):
    if mode not in set(['train', 'test']):
        print('mode should be train or test')
        raise Exception()

    x_image = tf.reshape(x, [-1, 28, 28, 1])
    paddings = tf.constant([[0, 0], [0, 1], [0, 1], [0, 0]], name='pad_const')
    x_image = tf.pad(x_image, paddings)

    W_conv1 = get_variable('W_conv1', [5, 5, 1, 5], mode)
    y = conv2d_stride_2_valid(x_image, W_conv1)
    W_bc1 = get_variable('W_conv1_bias', [1, 13, 13, 5], mode)
    y = y + W_bc1
    y = tf.nn.relu(y)

    y = avg_pool_3x3_same_size(y)
    W_conv2 = get_variable('W_conv2', [5, 5, 5, 50], mode)
    y = conv2d_stride_2_valid(y, W_conv2)
    y = avg_pool_3x3_same_size(y)

    y = tf.reshape(y, [-1, 5 * 5 * 50])
    W_fc1 = get_variable('W_fc1', [5 * 5 * 50, 100], mode)
    W_b1 = get_variable('W_fc1_bias', [100], mode)
    y = tf.matmul(y, W_fc1)
    y = y + W_b1
    y = tf.nn.relu(y)

    W_fc2 = get_variable('W_fc2', [100, 10], mode)
    W_b2 = get_variable('W_fc2_bias', [10], mode)
    y = tf.matmul(y, W_fc2)
    y + y + W_b2
    return y