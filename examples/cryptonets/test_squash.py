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
"""A simplified deep MNIST classifier using convolutional layers.
This script has the following changes when compared to mnist_deep.py:
1. no dropout layer (which disables the rng op)
2. no truncated normal initialzation(which disables the while op)

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile
import getpass
import time
import numpy as np
import itertools
import os
import glob

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import common

FLAGS = None


def cryptonets_test_squashed(x):
    """Constructs test network for Cryptonets using saved weights.
       Assumes linear layers have been squashed."""

    # Reshape to use within a convolutional neural net.
    # Last dimension is for "features" - there is only one here, since images are
    # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, 28, 28, 1])

    # First conv layer: maps one grayscale image to 5 feature maps of 13 x 13
    with tf.name_scope('conv1'):
        W_conv1 = tf.constant(
            np.loadtxt('W_conv1.txt', dtype=np.float32).reshape([5, 5, 1, 5]))
        h_conv1_no_pad = tf.square(
            common.conv2d_stride_2_valid(x_image, W_conv1))
        paddings = tf.constant([[0, 0], [0, 1], [0, 1], [0, 0]],
                               name='pad_const')
        h_conv1 = tf.pad(h_conv1_no_pad, paddings)

    with tf.name_scope('squash'):
        W_squash = tf.constant(
            np.loadtxt("W_squash.txt",
                       dtype=np.float32).reshape([5 * 13 * 13, 100]))

    with tf.name_scope('fc1'):
        h_pool2_flat = tf.reshape(h_conv1, [-1, 5 * 13 * 13])
        h_fc1 = tf.square(tf.matmul(h_pool2_flat, W_squash))

    # Map the 100 features to 10 classes, one for each digit
    with tf.name_scope('fc2'):
        W_fc2 = tf.constant(
            np.loadtxt('W_fc2.txt', dtype=np.float32).reshape([100, 10]))
        y_conv = tf.matmul(h_fc1, W_fc2)
    return y_conv


def main(_):
    import ngraph

    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])

    # Build the graph for the deep net
    y_conv = cryptonets_test_squashed(x)

    with tf.Session() as sess:
        x_test = mnist.test.images[:FLAGS.batch_size]
        y_test = mnist.test.labels[:FLAGS.batch_size]
        y_conv_val = y_conv.eval(feed_dict={x: x_test, y_: y_test})
        print(y_conv_val)

    # Rename serialized graph
    try:
        serialized_graphs = glob.glob("tf_function_ngraph*.json")
        if os.environ['NGRAPH_ENABLE_SERIALIZE'] == "1" and len(serialized_graphs) == 1:
            src_path = serialized_graphs[0]
            dst_path = "mnist_cryptonets_batch_%s.json" % (FLAGS.batch_size,)
            print("Moving", src_path, "to", dst_path)
            os.rename(src_path, dst_path)
    except:
        print("Renaming serialized graph not successful")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/tmp/tensorflow/mnist/input_data',
        help='Directory where input data is stored')
    parser.add_argument(
        '--batch_size', type=int, default=1, help='Batch Size')
    parser.add_argument(
        '--test_image_count',
        type=int,
        default=None,
        help="Number of test images to evaluate on")

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
