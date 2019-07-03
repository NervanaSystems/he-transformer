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

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import common

FLAGS = None


def squash_layers():
    print("Squashing layers")
    tf.compat.v1.reset_default_graph()

    # Input from h_conv1 squaring
    x = tf.compat.v1.placeholder(tf.float32, [None, 13, 13, 5])

    # Pooling layer
    h_pool1 = common.avg_pool_3x3_same_size(x)  # To N x 13 x 13 x 5

    # Second convolution
    W_conv2 = np.loadtxt(
        'W_conv2.txt', dtype=np.float32).reshape([5, 5, 5, 50])
    h_conv2 = common.conv2d_stride_2_valid(h_pool1, W_conv2)

    # Second pooling layer.
    h_pool2 = common.avg_pool_3x3_same_size(h_conv2)

    # Fully connected layer 1
    # Input: N x 5 x 5 x 50
    # Output: N x 100
    W_fc1 = np.loadtxt(
        'W_fc1.txt', dtype=np.float32).reshape([5 * 5 * 50, 100])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 5 * 5 * 50])
    pre_square = tf.matmul(h_pool2_flat, W_fc1)

    with tf.compat.v1.Session() as sess:
        x_in = np.eye(13 * 13 * 5)
        x_in = x_in.reshape([13 * 13 * 5, 13, 13, 5])
        W = (sess.run([pre_square], feed_dict={x: x_in}))[0]
        squashed_file_name = "W_squash.txt"
        np.savetxt(squashed_file_name, W)
        print("Saved to", squashed_file_name)

        # Sanity check
        x_in = np.random.rand(100, 13, 13, 5)
        network_out = (sess.run([pre_square], feed_dict={x: x_in}))[0]
        linear_out = x_in.reshape(100, 13 * 13 * 5).dot(W)
        assert (np.max(np.abs(linear_out - network_out)) < 1e-5)

    print("Squashed layers")


def cryptonets_train(x):
    """Builds the graph for classifying digits based on Cryptonets

    Args:
        x: an input tensor with the dimensions (N_examples, 784), where 784 is
        the number of pixels in a standard MNIST image.

    Returns:
        A tuple (y, a scalar placeholder). y is a tensor of shape
        (N_examples, 10), with values equal to the logits of classifying the
        digit into one of 10 classes (the digits 0-9).
    """
    # Reshape to use within a conv neural net.
    # Last dimension is for "features" - there is only one here, since images
    # are grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
    with tf.name_scope('reshape'):
        paddings = [[0, 0], [0, 1], [0, 1], [0, 0]]
        x = tf.pad(x, paddings)

    # First conv layer
    # CryptoNets's output of the first conv layer has feature map size 13 x 13,
    # therefore, we manually add paddings.
    # Input: N x 28 x 28 x 1
    # Filter: 5 x 5 x 1 x 5
    # Output: N x 12 x 12 x 5
    # Output after padding: N x 13 x 13 x 5
    with tf.name_scope('conv1'):
        W_conv1 = tf.get_variable("W_conv1", [5, 5, 1, 5])
        h_conv1_no_pad = tf.square(common.conv2d_stride_2_valid(x, W_conv1))
        paddings = tf.constant([[0, 0], [0, 1], [0, 1], [0, 0]],
                               name='pad_const')
        h_conv1 = tf.pad(h_conv1_no_pad, paddings)

    # Pooling layer
    # Input: N x 13 x 13 x 5
    # Output: N x 13 x 13 x 5
    with tf.name_scope('pool1'):
        h_pool1 = common.avg_pool_3x3_same_size(h_conv1)

    # Second convolution
    # Input: N x 13 x 13 x 5
    # Filter: 5 x 5 x 5 x 50
    # Output: N x 5 x 5 x 50
    with tf.name_scope('conv2'):
        W_conv2 = tf.get_variable("W_conv2", [5, 5, 5, 50])
        h_conv2 = common.conv2d_stride_2_valid(h_pool1, W_conv2)

    # Second pooling layer
    # Input: N x 5 x 5 x 50
    # Output: N x 5 x 5 x 50
    with tf.name_scope('pool2'):
        h_pool2 = common.avg_pool_3x3_same_size(h_conv2)

    # Fully connected layer 1
    # Input: N x 5 x 5 x 50
    # Input flattened: N x 1250
    # Weight: 1250 x 100
    # Output: N x 100
    with tf.name_scope('fc1'):
        h_pool2_flat = tf.reshape(h_pool2, [-1, 5 * 5 * 50])
        W_fc1 = tf.get_variable("W_fc1", [5 * 5 * 50, 100])
        h_fc1 = tf.square(tf.matmul(h_pool2_flat, W_fc1))

    # Map the 100 features to 10 classes, one for each digit
    # Input: N x 100
    # Weight: 100 x 10
    # Output: N x 10
    with tf.name_scope('fc2'):
        W_fc2 = tf.get_variable("W_fc2", [100, 10])
        y_conv = tf.matmul(h_fc1, W_fc2)
    return y_conv


def get_train_batch(train_iter, batch_size, x_train, y_train):
    start_index = train_iter * batch_size
    end_index = start_index + batch_size

    data_count = x_train.shape[0]

    if start_index > data_count and end_index > data_count:
        start_index %= data_count
        end_index %= data_count
        x_batch = x_train[start_index:end_index]
        y_batch = y_train[start_index:end_index]
    elif end_index > data_count:
        end_index %= data_count
        x_batch = np.concatenate((x_train[start_index:], x_train[0:end_index]))
        y_batch = np.concatenate((y_train[start_index:], y_train[0:end_index]))
    else:
        x_batch = x_train[start_index:end_index]
        y_batch = y_train[start_index:end_index]

    return x_batch, y_batch


def main(_):
    (x_train, y_train, x_test, y_test) = common.load_mnist_data()

    print('x_train', x_train.shape)
    print('x-test', x_test.shape)

    x = tf.compat.v1.placeholder(tf.float32, [None, 28, 28, 1])
    y_ = tf.compat.v1.placeholder(tf.float32, [None, 10])
    y_conv = common.cryptonets_model(x, 'train')

    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=y_, logits=y_conv)
    cross_entropy = tf.reduce_mean(cross_entropy)

    with tf.name_scope('adam_optimizer'):
        train_step = tf.compat.v1.train.AdamOptimizer(1e-4).minimize(
            cross_entropy)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        loss_values = []
        for i in range(FLAGS.train_loop_count):
            x_batch, y_batch = get_train_batch(i, FLAGS.batch_size, x_train,
                                               y_train)
            if i % 100 == 0:
                t = time.time()
                train_accuracy = accuracy.eval(feed_dict={
                    x: x_batch,
                    y_: y_batch
                })
                print('step %d, training accuracy %g, %g msec to evaluate' %
                      (i, train_accuracy, 1000 * (time.time() - t)))
            t = time.time()
            _, loss = sess.run([train_step, cross_entropy],
                               feed_dict={
                                   x: x_batch,
                                   y_: y_batch
                               })
            loss_values.append(loss)

            if i % 1000 == 999 or i == FLAGS.train_loop_count - 1:
                test_accuracy = accuracy.eval(feed_dict={
                    x: x_test,
                    y_: y_test
                })
                print('test accuracy %g' % test_accuracy)

        print("Training finished. Saving variables.")
        for var in tf.compat.v1.get_collection(
                tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES):
            weight = (sess.run([var]))[0].flatten().tolist()
            filename = (str(var).split())[1].replace('/', '_')
            filename = filename.replace("'", "").replace(':0', '') + '.txt'

            print("saving", filename)
            np.savetxt(str(filename), weight)

    # Squash weights and save as W_squash.txt
    squash_layers()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/tmp/tensorflow/mnist/input_data',
        help='Directory where input data is stored')
    parser.add_argument(
        '--train_loop_count',
        type=int,
        default=20000,
        help='Number of training iterations')
    parser.add_argument(
        '--batch_size', type=int, default=50, help='Batch Size')
    parser.add_argument(
        '--test_image_count',
        type=int,
        default=None,
        help="Number of test images to evaluate on")
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)