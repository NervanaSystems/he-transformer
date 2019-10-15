# ==============================================================================
#  Copyright 2018-2019 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ==============================================================================
"""An MNIST classifier based on Cryptonets using convolutional layers. """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import time
import numpy as np
import itertools
import glob
import tensorflow as tf
import ngraph_bridge
import os
from tensorflow.core.protobuf import rewriter_config_pb2

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mnist_util import load_mnist_data, \
                       get_variable, \
                       conv2d_stride_2_valid, \
                       str2bool, \
                       server_argument_parser, \
                       server_config_from_flags


def cryptonets_test_squashed(x):
    """Constructs test network for Cryptonets using saved weights.
       Assumes linear layers have been squashed."""
    paddings = [[0, 0], [0, 1], [0, 1], [0, 0]]
    x = tf.pad(x, paddings)

    W_conv1 = get_variable('W_conv1', [5, 5, 1, 5], 'test')
    y = conv2d_stride_2_valid(x, W_conv1)
    y = tf.square(y)
    W_squash = get_variable('W_squash', [5 * 13 * 13, 100], 'test')
    y = tf.reshape(y, [-1, 5 * 13 * 13])
    y = tf.matmul(y, W_squash)
    y = tf.square(y)
    W_fc2 = get_variable('W_fc2', [100, 10], 'test')
    y = tf.matmul(y, W_fc2)
    return y


def test_mnist_cnn(FLAGS):
    (x_train, y_train, x_test, y_test) = load_mnist_data()

    x = tf.compat.v1.placeholder(tf.float32, [None, 28, 28, 1], name='input')
    y_ = tf.compat.v1.placeholder(tf.float32, [None, 10])

    # Create the model
    y_conv = cryptonets_test_squashed(x)

    config = server_config_from_flags(FLAGS, x.name)

    print('config', config)

    with tf.compat.v1.Session(config=config) as sess:
        x_test = x_test[:FLAGS.batch_size]
        y_test = y_test[:FLAGS.batch_size]
        start_time = time.time()
        y_conv_val = y_conv.eval(feed_dict={x: x_test, y_: y_test})
        elasped_time = (time.time() - start_time)
        print("total time(s)", np.round(elasped_time, 3))
        print('y_conv_val', np.round(y_conv_val, 2))

    y_test_batch = y_test[:FLAGS.batch_size]
    y_label_batch = np.argmax(y_test_batch, 1)

    correct_prediction = np.equal(np.argmax(y_conv_val, 1), y_label_batch)
    error_count = np.size(correct_prediction) - np.sum(correct_prediction)
    test_accuracy = np.mean(correct_prediction)

    print('Error count:', error_count, 'of', FLAGS.batch_size, 'elements.')
    print('Accuracy: ', test_accuracy)


if __name__ == '__main__':
    parser = server_argument_parser()
    FLAGS, unparsed = parser.parse_known_args()

    if unparsed:
        print('Unparsed flags:', unparsed)
    if FLAGS.encrypt_server_data and FLAGS.enable_client:
        raise Exception(
            "encrypt_server_data flag only valid when client is not enabled. Note: the client can specify whether or not to encrypt the data using 'encrypt' or 'plain' in the configuration map"
        )

    test_mnist_cnn(FLAGS)