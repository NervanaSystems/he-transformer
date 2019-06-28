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


def conv2d_stride_2_valid(x, W, name=None):
    """returns a 2d convolution layer with stride 2, valid pooling"""
    return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='VALID')


def avg_pool_3x3_same_size(x):
    """3x3 avg_pool using same padding, keeping original feature map size"""
    return tf.nn.avg_pool(
        x, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')


def get_variable(name, shape, mode):
    if mode not in set(['train', 'test']):
        print('mode should be train or test')
        raise Exception()

    if mode == 'train':
        return tf.get_variable(name, shape)
    else:
        print('loading ', name)
        return tf.constant(
            np.loadtxt(name + '.txt', dtype=np.float32).reshape(shape))


def cryptonets_model(x, mode):
    if mode not in set(['train', 'test']):
        print('mode should be train or test')
        raise Exception()

    x_image = tf.reshape(x, [-1, 28, 28, 1])

    # Conv
    W_conv1 = get_variable("W_conv1", [5, 5, 1, 5], mode)
    y = conv2d_stride_2_valid(x_image, W_conv1)
    y = tf.square(y)
    y = tf.reshape(y, [-1, 5 * 12 * 12])

    # FC
    W_fc1 = get_variable("W_fc1", [5 * 12 * 12, 100], mode)
    y = tf.matmul(y, W_fc1)
    y = tf.square(y)

    # FC
    W_fc2 = get_variable("W_fc2", [100, 10], mode)
    y = tf.matmul(y, W_fc2)

    return y