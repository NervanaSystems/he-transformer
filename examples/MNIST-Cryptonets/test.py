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
import common
import ngraph_bridge

import os
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
        h_fc1 = tf.matmul(h_pool2_flat, W_squash)
        # h_fc1 = tf.Print(h_fc1, [h_fc1], summarize=200,  message="After dot\n")
        h_fc1 = tf.square(h_fc1)

    # Map the 100 features to 10 classes, one for each digit
    with tf.name_scope('fc2'):
        W_fc2 = tf.constant(
            np.loadtxt('W_fc2.txt', dtype=np.float32).reshape([100, 10]))
        y_conv = tf.matmul(h_fc1, W_fc2)
        y_conv = tf.Print(y_conv, [y_conv], summarize=20, message="Result\n")
    return y_conv


def cryptonets_test_original(x):
    """Constructs test network for Cryptonets using saved weights"""

    # Reshape to use within a convolutional neural net.
    # Last dimension is for "features" - there is only one here, since images
    # are grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, 28, 28, 1])

    # First conv layer - maps one grayscale image to 5 feature maps of 13 x 13
    with tf.name_scope('conv1'):
        W_conv1 = tf.constant(
            np.loadtxt('W_conv1.txt', dtype=np.float32).reshape([5, 5, 1, 5]))
        h_conv1_no_pad = tf.square(
            common.conv2d_stride_2_valid(x_image, W_conv1))
        paddings = tf.constant([[0, 0], [0, 1], [0, 1], [0, 0]],
                               name='pad_const')
        h_conv1 = tf.pad(h_conv1_no_pad, paddings)

    # Pooling layer
    with tf.name_scope('pool1'):
        h_pool1 = common.avg_pool_3x3_same_size(h_conv1)  # To 5 x 13 x 13

    # Second convolution
    with tf.name_scope('conv2'):
        W_conv2 = tf.constant(
            np.loadtxt('W_conv2.txt', dtype=np.float32).reshape([5, 5, 5, 50]))
        h_conv2 = common.conv2d_stride_2_valid(h_pool1, W_conv2)

    # Second pooling layer.
    with tf.name_scope('pool2'):
        h_pool2 = common.avg_pool_3x3_same_size(h_conv2)

    # Fully connected layer 1
    # Input: N x 5 x 5 x 50
    # Output: N x 100
    with tf.name_scope('fc1'):
        W_fc1 = tf.constant(
            np.loadtxt('W_fc1.txt',
                       dtype=np.float32).reshape([5 * 5 * 50, 100]))
        h_pool2_flat = tf.reshape(h_pool2, [-1, 5 * 5 * 50])
        h_fc1 = tf.square(tf.matmul(h_pool2_flat, W_fc1))

    # Map the 100 features to 10 classes, one for each digit
    with tf.name_scope('fc2'):
        W_fc2 = tf.constant(
            np.loadtxt('W_fc2.txt', dtype=np.float32).reshape([100, 10]))
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
        y_conv = cryptonets_test_original(x)
    else:
        y_conv = cryptonets_test_squashed(x)

    x_test = mnist.test.images[:FLAGS.batch_size]
    y_test = mnist.test.labels[:FLAGS.batch_size]
    x_test_batch = mnist.test.images[:FLAGS.batch_size]
    y_test_batch = mnist.test.labels[:FLAGS.batch_size]

    # Run warm-up and 10 trials
    with tf.Session() as sess:
        start_time = time.time()
        # Run model
        y_conv_val = y_conv.eval(feed_dict={x: x_test, y_: y_test})
        elasped_time = time.time() - start_time
        print("total time warmup:", elasped_time)

    # Avoid performing in a session, to allow he backends to report accuracy.
    if FLAGS.report_accuracy:
        y_label_batch = np.argmax(y_test_batch, 1)
        correct_prediction = np.equal(np.argmax(y_conv_val, 1), y_label_batch)
        error_count = np.size(correct_prediction) - np.sum(correct_prediction)
        test_accuracy = np.mean(correct_prediction)

        print('Error count', error_count, 'of', FLAGS.batch_size, 'elements.')
        print('Accuracy with ' + network + ': %g ' % test_accuracy)

    with tf.Session() as sess:
        start_time = time.time()
        # Run model
        y_conv_val = y_conv.eval(feed_dict={x: x_test, y_: y_test})
        elasped_time = time.time() - start_time
        print("total time trial 1:", elasped_time)

    with tf.Session() as sess:
        start_time = time.time()
        # Run model
        y_conv_val = y_conv.eval(feed_dict={x: x_test, y_: y_test})
        elasped_time = time.time() - start_time
        print("total time trial 2:", elasped_time)
    with tf.Session() as sess:
        start_time = time.time()
        # Run model
        y_conv_val = y_conv.eval(feed_dict={x: x_test, y_: y_test})
        elasped_time = time.time() - start_time
        print("total time trial 3:", elasped_time)
    with tf.Session() as sess:
        start_time = time.time()
        # Run model
        y_conv_val = y_conv.eval(feed_dict={x: x_test, y_: y_test})
        elasped_time = time.time() - start_time
        print("total time trial 4:", elasped_time)
    with tf.Session() as sess:
        start_time = time.time()
        # Run model
        y_conv_val = y_conv.eval(feed_dict={x: x_test, y_: y_test})
        elasped_time = time.time() - start_time
        print("total time trial 5:", elasped_time)
    with tf.Session() as sess:
        start_time = time.time()
        # Run model
        y_conv_val = y_conv.eval(feed_dict={x: x_test, y_: y_test})
        elasped_time = time.time() - start_time
        print("total time trial 6:", elasped_time)
    with tf.Session() as sess:
        start_time = time.time()
        # Run model
        y_conv_val = y_conv.eval(feed_dict={x: x_test, y_: y_test})
        elasped_time = time.time() - start_time
        print("total time trial 7:", elasped_time)
    with tf.Session() as sess:
        start_time = time.time()
        # Run model
        y_conv_val = y_conv.eval(feed_dict={x: x_test, y_: y_test})
        elasped_time = time.time() - start_time
        print("total time trial 8:", elasped_time)
    with tf.Session() as sess:
        start_time = time.time()
        # Run model
        y_conv_val = y_conv.eval(feed_dict={x: x_test, y_: y_test})
        elasped_time = time.time() - start_time
        print("total time trial 9:", elasped_time)
    with tf.Session() as sess:
        start_time = time.time()
        # Run model
        y_conv_val = y_conv.eval(feed_dict={x: x_test, y_: y_test})
        elasped_time = time.time() - start_time
        print("total time trial 10:", elasped_time)

    if FLAGS.save_batch:
        x_test_batch.tofile("x_test_" + str(FLAGS.batch_size) + ".bin")
        y_label_batch.astype('float32').tofile("y_label_" +
                                               str(FLAGS.batch_size) + ".bin")

    # Avoid performing in a session, to allow he backends to report accuracy.
    if FLAGS.report_accuracy:
        correct_prediction = np.equal(np.argmax(y_conv_val, 1), y_label_batch)
        error_count = np.size(correct_prediction) - np.sum(correct_prediction)
        test_accuracy = np.mean(correct_prediction)

        print('Error count', error_count, 'of', FLAGS.batch_size, 'elements.')
        print('Accuracy with ' + network + ': %g ' % test_accuracy)

    # Rename serialized graph
    try:
        serialized_graphs = glob.glob("tf_function_ngraph*.json")
        if os.environ.get('NGRAPH_ENABLE_SERIALIZE',
                          '') == "1" and len(serialized_graphs) == 1:
            src_path = serialized_graphs[0]
            dst_path = "mnist_cryptonets_batch_%s.json" % (FLAGS.batch_size, )
            print("Moving", src_path, "to", dst_path)
            os.rename(src_path, dst_path)
    except:
        print("Renaming serialized graph not successful")


def main(_):
    # Disable mnist dataset deprecation warning
    tf.logging.set_verbosity(tf.logging.ERROR)

    # Test using the original graph
    # test_mnist_cnn(FLAGS, 'orig')

    # Test using squashed graph
    test_mnist_cnn(FLAGS, 'squash')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/tmp/tensorflow/mnist/input_data',
        help='Directory where input data is stored')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument(
        '--test_image_count',
        type=int,
        default=None,
        help="Number of test images to evaluate on")
    parser.add_argument(
        '--save_batch',
        type=bool,
        default=False,
        help='Whether or not to save the test image and label.')
    parser.add_argument(
        '--report_accuracy',
        type=bool,
        default=False,
        help='Whether or not to save the compute the test accuracy.')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
