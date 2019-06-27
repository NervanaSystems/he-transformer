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

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import ngraph_bridge

import os
FLAGS = None


def conv_layer(x):
    """Constructs convolution layer for MNIST network"""

    # Reshape to use within a convolutional neural net.
    # Last dimension is for "features" - there is only one here, since images
    # are grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, 28, 28, 1])

    # First conv layer - maps one grayscale image to 5 feature maps of 13 x 13
    with tf.name_scope('conv1'):
        W_conv1 = tf.constant(
            np.loadtxt('W_conv1.txt', dtype=np.float32).reshape([5, 5, 1, 5]))
        h_conv = tf.nn.conv2d(
            x_image, W_conv1, strides=[1, 2, 2, 1], padding='VALID')

    return h_conv


def test_conv(FLAGS):
    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    x = tf.placeholder(tf.float32, [None, 784])
    y_conv = conv_layer(x)

    with tf.Session() as sess:
        x_test = mnist.test.images[:FLAGS.batch_size]
        start_time = time.time()
        y_conv_val = y_conv.eval(feed_dict={x: x_test})

        print('y_conv_val', y_conv_val.shape)
        elasped_time = (time.time() - start_time)
        print("total time(s)", np.round(elasped_time, 3))

    x_test_batch = mnist.test.images[:FLAGS.batch_size]
    y_test_batch = mnist.test.labels[:FLAGS.batch_size]
    x_test = mnist.test.images
    y_test = mnist.test.labels

    y_label_batch = np.argmax(y_test_batch, 1)


def main(_):
    # Disable mnist dataset deprecation warning
    tf.logging.set_verbosity(tf.logging.ERROR)

    test_conv(FLAGS)


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