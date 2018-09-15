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
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

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

FLAGS = None


def conv2d(x, W, name=None):
    """conv2d returns a 2d convolution layer with stride 2."""
    return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME')


def scaled_mean_pool_2x2(x):
    """scaled_mean_pool_2x keeps feature map size."""
    return tf.nn.avg_pool(
        x, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')


def weight_variable(shape, name):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.get_variable(name, shape)
    return tf.Variable(initial)


def deepnn(x):
    """Builds the graph for classifying digits based on Cryptonets

    Args:
        x: an input tensor with the dimensions (N_examples, 784), where 784 is
        the number of pixels in a standard MNIST image.

    Returns:
        A tuple (y, a scalar placeholder). y is a tensor of shape
        (N_examples, 10), with values equal to the logits of classifying the
        digit into one of 10 classes (the digits 0-9). The scalar placeholder is
        meant for the probability of dropout. Since we don't use a dropout layer
        in this script, this placeholder is of no relavance and acts as a dummy.
    """
    # Reshape to use within a conv neural net.
    # Last dimension is for "features" - there is only one here, since images
    # are grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, 28, 28, 1])

    # First conv layer - maps one grayscale image to 5 feature maps of 14x14
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([5, 5, 1, 5], "W_conv1")
        h_conv1 = tf.square(conv2d(x_image, W_conv1))

    # Pooling layer
    with tf.name_scope('pool1'):
        h_pool1 = scaled_mean_pool_2x2(h_conv1)  # To 5x14x14

    # Second convolution
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([5, 5, 5, 50], "W_conv2")  # To 50x7x7
        h_conv2 = conv2d(h_pool1, W_conv2)

    # Second pooling layer.
    with tf.name_scope('pool2'):
        h_pool2 = scaled_mean_pool_2x2(h_conv2)

    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # is down to 7x7x11 feature maps -- maps this to 100 features.
    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([7 * 7 * 50, 100], "W_fc1")
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 50])
        h_fc1 = tf.square(tf.matmul(h_pool2_flat, W_fc1))

    # Map the 100 features to 10 classes, one for each digit
    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([100, 10], "W_fc2")
        y_conv = tf.matmul(h_fc1, W_fc2)
    return y_conv


def main(_):
    # Config
    config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False,
        inter_op_parallelism_threads=1)

    # Note: Additional configuration option to boost performance is to set the
    # following environment for the run:
    # OMP_NUM_THREADS=44 KMP_AFFINITY=granularity=fine,scatter
    # The OMP_NUM_THREADS number should correspond to the number of
    # cores in the system

    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])

    # Build the graph for the deep net
    y_conv = deepnn(x)

    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=y_, logits=y_conv)
    cross_entropy = tf.reduce_mean(cross_entropy)

    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)
    tf.summary.scalar('Training accuracy', accuracy)
    tf.summary.scalar('Loss function', cross_entropy)

    graph_location = "/tmp/" + getpass.getuser(
    ) + "/tensorboard-logs/mnist-convnet"
    print('Saving graph to: %s' % graph_location)

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(graph_location)
    train_writer.add_graph(tf.get_default_graph())

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        train_loops = FLAGS.train_loop_count
        loss_values = []
        for i in range(train_loops):
            batch = mnist.train.next_batch(FLAGS.batch_size)
            if i % 100 == 0:
                t = time.time()
                train_accuracy = accuracy.eval(feed_dict={
                    x: batch[0],
                    y_: batch[1]
                })
                #tf.summary.scalar('Training accuracy', train_accuracy)
                print('step %d, training accuracy %g, %g sec to evaluate' %
                      (i, train_accuracy, time.time() - t))
            t = time.time()
            _, summary, loss = sess.run(
                [train_step, merged, cross_entropy],
                feed_dict={
                    x: batch[0],
                    y_: batch[1]
                })
            loss_values.append(loss)
            # print('step %d, loss %g, %g sec for training step'
            #       % (i, loss, time.time() - t ))
            train_writer.add_summary(summary, i)

            if i % 1000 == 999 or i == train_loops - 1:

                num_test_images = FLAGS.test_image_count
                x_test = mnist.test.images[:num_test_images]
                y_test = mnist.test.labels[:num_test_images]

                test_accuracy = accuracy.eval(feed_dict={
                    x: x_test,
                    y_: y_test
                })
                print('test accuracy %g' % test_accuracy)

                if i == train_loops - 1:
                    print("Training finished. Saving variables")
                    for var in tf.get_collection(
                            tf.GraphKeys.TRAINABLE_VARIABLES):
                        weight = (sess.run([var]))[0].flatten().tolist()
                        filename = (str(var).split())[1].replace('/', '_')
                        filename = filename.replace("'", "").replace(
                            ':0', '') + '.txt'
                        # Don't save initial variable weights
                        if filename not in set([
                                'W_conv1.txt', 'W_conv2.txt', 'W_fc1.txt',
                                'W_fc2.txt'
                        ]):
                            print("saving", filename)
                            np.savetxt(str(filename), weight)

                    return loss_values, test_accuracy


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
