# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""An MNIST classifier using convolutional layers and relu activations. """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import time
import numpy as np
import itertools
import glob

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import common
from common import get_variable, conv2d_stride_2_valid
import ngraph_bridge

import os
FLAGS = None


def cryptonets_test_relu_squashed(x):
    """Constructs test network for Cryptonets using saved weights.
       Assumes linear layers have been squashed."""
    x = tf.reshape(x, [-1, 28, 28, 1])
    paddings = tf.constant([[0, 0], [0, 1], [0, 1], [0, 0]], name='pad_const')
    x = tf.pad(x, paddings)

    W_conv1 = get_variable('W_conv1', [5, 5, 1, 5], 'test')
    y = conv2d_stride_2_valid(x, W_conv1)
    W_bc1 = get_variable('W_conv1_bias', [1, 13, 13, 5], 'test')
    y = y + W_bc1
    y = tf.nn.relu(y)

    W_squash = get_variable('W_squash', [5 * 13 * 13, 100], 'test')
    y = tf.reshape(y, [-1, 5 * 13 * 13])
    y = tf.matmul(y, W_squash)
    W_b1 = get_variable('W_fc1_bias', [100], 'test')
    y = y + W_b1

    y = tf.nn.relu(y)
    W_fc2 = get_variable('W_fc2', [100, 10], 'test')
    y = tf.matmul(y, W_fc2)

    W_b2 = get_variable('W_fc2_bias', [10], 'test')
    y = y + W_b2

    return y


def test_cryptonets_relu(FLAGS):

    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])

    y_conv = cryptonets_test_relu_squashed(x)

    with tf.Session() as sess:
        start_time = time.time()
        x_test = mnist.test.images[:FLAGS.batch_size]
        y_test = mnist.test.labels[:FLAGS.batch_size]

        # Run model
        y_conv_val = y_conv.eval(feed_dict={x: x_test, y_: y_test})
        elasped_time = time.time() - start_time
        print("total time(s)", np.round(elasped_time, 2))
        print('result')
        print(np.round(y_conv_val, 2))

    x_test_batch = mnist.test.images[:FLAGS.batch_size]
    y_test_batch = mnist.test.labels[:FLAGS.batch_size]
    x_test = mnist.test.images
    y_test = mnist.test.labels

    y_label_batch = np.argmax(y_test_batch, 1)

    y_pred = np.argmax(y_conv_val, 1)
    print('y_pred', y_pred)
    correct_prediction = np.equal(y_pred, y_label_batch)
    error_count = np.size(correct_prediction) - np.sum(correct_prediction)
    test_accuracy = np.mean(correct_prediction)

    print('Error count', error_count, 'of', FLAGS.batch_size, 'elements.')
    print('Accuracy: %g ' % test_accuracy)


def main(_):
    # Disable mnist dataset deprecation warning
    tf.logging.set_verbosity(tf.logging.ERROR)

    test_cryptonets_relu(FLAGS)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/tmp/tensorflow/mnist/input_data',
        help='Directory where input data is stored')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
