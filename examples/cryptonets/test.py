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

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import common

FLAGS = None


def squash_layers():
    print("Squashing layers")

    tf.reset_default_graph()

    # Input from h_conv1 squaring
    x = tf.placeholder(tf.float32, [None, 13, 13, 5])

    # Pooling layer
    h_pool1 = common.avg_pool_3x3_same_size(x)  # To N x 13 x 13 x 5

    # Second convolution
    W_conv2 = np.loadtxt(
        'W_conv2.txt', dtype=np.float32).reshape([5, 5, 5, 50])
    h_conv2 = common.conv2d_stride_2_valid(h_pool1, W_conv2)

    # Second pooling layer.
    h_pool2 = common.avg_pool_3x3_same_size(h_conv2)

    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # is down to 7x7x11 feature maps -- maps this to 100 features.
    W_fc1 = np.loadtxt(
        'W_fc1.txt', dtype=np.float32).reshape([5 * 5 * 50, 100])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 5 * 5 * 50])
    pre_square = tf.matmul(h_pool2_flat, W_fc1)

    with tf.Session() as sess:
        x_in = np.eye(13 * 13 * 5)
        x_in = x_in.reshape([13 * 13 * 5, 13, 13, 5])
        W = (sess.run([pre_square], feed_dict={x: x_in}))[0]
        np.savetxt("W_squash.txt", W)

        # Sanity check
        x_in = np.random.rand(100, 13, 13, 5)
        network_out = (sess.run([pre_square], feed_dict={x: x_in}))[0]
        linear_out = x_in.reshape(100, 13 * 13 * 5).dot(W)
        assert (np.max(np.abs(linear_out - network_out)) < 1e-5)

    print("Squashed layers")


def test_deepnn(x):
    """Constructs test network for Cryptonets using saved weights.
       Assumes linear layers have been squashed."""

    # Reshape to use within a convolutional neural net.
    # Last dimension is for "features" - there is only one here, since images are
    # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, 28, 28, 1])

    # First conv layer: maps one grayscale image to 5 feature maps of 14x14
    with tf.name_scope('conv1'):
        W_conv1 = np.loadtxt(
            'W_conv1.txt', dtype=np.float32).reshape([5, 5, 1, 5])
        h_conv1 = tf.square(common.conv2d_stride_2_valid(x_image, W_conv1))

    with tf.name_scope('squash'):
        W_squash = np.loadtxt(
            "W_squash.txt", dtype=np.float32).reshape([5 * 14 * 14, 100])

    with tf.name_scope('fc1'):
        h_pool2_flat = tf.reshape(h_conv1, [-1, 5 * 14 * 14])
        h_fc1 = tf.square(tf.matmul(h_pool2_flat, W_squash))

    # Map the 100 features to 10 classes, one for each digit
    with tf.name_scope('fc2'):
        W_fc2 = np.loadtxt('W_fc2.txt', dtype=np.float32).reshape([100, 10])
        y_conv = tf.matmul(h_fc1, W_fc2)
    return y_conv


def test_deepnn_orig(x):
    """Constructs test network for Cryptonets using saved weights"""

    # Reshape to use within a convolutional neural net.
    # Last dimension is for "features" - there is only one here, since images
    # are grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, 28, 28, 1])

    # First conv layer - maps one grayscale image to 5 feature maps of 14x14
    with tf.name_scope('conv1'):
        W_conv1 = np.loadtxt(
            'W_conv1.txt', dtype=np.float32).reshape([5, 5, 1, 5])
        h_conv1 = tf.square(common.conv2d_stride_2_valid(x_image, W_conv1))

    # Pooling layer
    with tf.name_scope('pool1'):
        h_pool1 = common.avg_pool_3x3_same_size(h_conv1)  # To 5x14x14

    # Second convolution
    with tf.name_scope('conv2'):
        W_conv2 = np.loadtxt(
            'W_conv2.txt', dtype=np.float32).reshape([5, 5, 5, 50])
        h_conv2 = common.conv2d_stride_2_valid(h_pool1, W_conv2)

    # Second pooling layer.
    with tf.name_scope('pool2'):
        h_pool2 = common.avg_pool_3x3_same_size(h_conv2)

    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # is down to 7x7x11 feature maps -- maps this to 100 features.
    with tf.name_scope('fc1'):
        W_fc1 = np.loadtxt(
            'W_fc1.txt', dtype=np.float32).reshape([7 * 7 * 50, 100])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 50])
        h_fc1 = tf.square(tf.matmul(h_pool2_flat, W_fc1))

    # Map the 100 features to 10 classes, one for each digit
    with tf.name_scope('fc2'):
        W_fc2 = np.loadtxt('W_fc2.txt', dtype=np.float32).reshape([100, 10])
        y_conv = tf.matmul(h_fc1, W_fc2)
    return y_conv


def test_mnist_cnn(FLAGS, network):
    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])

    # Build the graph for the deep net
    if network == 'orig':
        y_conv = test_deepnn_orig(x)
    else:
        y_conv = test_deepnn(x)

    with tf.Session() as sess:
        batch_size = 1
        x_test = mnist.test.images[:batch_size]
        y_test = mnist.test.labels[:batch_size]
        y_conv_val = y_conv.eval(feed_dict={x: x_test, y_: y_test})
        print(y_conv_val)

    # with tf.name_scope('accuracy'):
    #     correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    #     correct_prediction = tf.cast(correct_prediction, tf.float32)
    # accuracy = tf.reduce_mean(correct_prediction)

    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())

    #     num_test_images = FLAGS.test_image_count
    #     x_test = mnist.test.images[:num_test_images]
    #     y_test = mnist.test.labels[:num_test_images]

    #     test_accuracy = accuracy.eval(feed_dict={x: x_test, y_: y_test})
    #     print('test accuracy wth ' + network + ': %g' % test_accuracy)

    # Run again to export inference graph on smaller batch size
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())

    #     batch_size = 1
    #     x_test = mnist.test.images[:batch_size]
    #     y_test = mnist.test.labels[:batch_size]

    #     y_label = np.argmax(y_test, 1)

    #     x_test.tofile("x_test_" + str(batch_size) + ".bin")
    #     y_test.astype('float32').tofile("y_test_" + str(batch_size) + ".bin")
    #     y_label.astype('float32').tofile("y_label_" + str(batch_size) + ".bin")

    #     test_accuracy = accuracy.eval(feed_dict={x: x_test, y_: y_test})
    #     print('test accuracy wth ' + network + ': %g' % test_accuracy)


def main(_):
    # Disable mnist dataset deprecation warning
    tf.logging.set_verbosity(tf.logging.ERROR)

    # Squash layer and write weights
    squash_layers()

    # Test using the original graph
    # test_mnist_cnn(FLAGS, 'orig')

    # Test using squashed graph
    # test_mnist_cnn(FLAGS, 'squash')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/tmp/tensorflow/mnist/input_data',
        help='Directory where input data is stored')
    parser.add_argument(
        '--batch_size', type=int, default=50, help='Batch Size')
    parser.add_argument(
        '--test_image_count',
        type=int,
        default=None,
        help="Number of test images to evaluate on")

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
