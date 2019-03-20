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
from common import SCALING, NUM_KERNELS, FC1_SIZE, FC2_SIZE

FLAGS = None


# Forward pass = bai
# Backward pass = dy.
@tf.custom_gradient
def straight_through_estimator(x):
    def grad(dy):
        return dy

    return tf.sign(x), grad


def load_variable(filename, shape):
    return tf.constant(
        np.loadtxt(filename + '.txt', dtype=np.float32).reshape(shape))


def cryptonets_train(x, is_training):
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, 28, 28, 1])

    with tf.name_scope('conv1'):
        W_conv1 = tf.get_variable("W_conv1", [5, 5, 1, NUM_KERNELS])
        W_conv1 = tf.clip_by_value(W_conv1, -1, 1)
        W_conv1 = straight_through_estimator(W_conv1)
        h_conv1 = common.conv2d_stride_2_valid(x_image, W_conv1)
        paddings = tf.constant([[0, 0], [0, 1], [0, 1], [0, 0]],
                               name='pad_const')
        h_conv1 = tf.pad(h_conv1, paddings)
        h_conv1 = tf.reshape(h_conv1, [-1, FC1_SIZE])

        # use object version to get variables
        #bn_instance = tf.layers.BatchNormalization(trainable=True)
        #h_conv1 = bn_instance.apply(h_conv1)
        h_conv1 = tf.layers.batch_normalization(h_conv1, training=is_training)

        h_conv1 = tf.square(h_conv1)

    with tf.name_scope('fc1'):
        W_fc1 = tf.get_variable("W_fc1", [FC1_SIZE, FC2_SIZE])
        W_fc1 = tf.clip_by_value(W_fc1, -1, 1)
        W_fc1 = straight_through_estimator(W_fc1)
        h_fc1 = tf.matmul(h_conv1, W_fc1)

        h_fc1 = tf.layers.batch_normalization(h_fc1, training=is_training)
        h_fc1 = tf.square(h_fc1)

        h_fc1 = tf.reshape(h_fc1, [-1, FC2_SIZE])

    with tf.name_scope('fc2'):
        W_fc2 = tf.get_variable("W_fc2", [FC2_SIZE, 10])
        W_fc2 = tf.clip_by_value(W_fc2, -1, 1)
        W_fc2 = straight_through_estimator(W_fc2)
        y_conv = tf.matmul(h_fc1, W_fc2)

        y_conv = tf.layers.batch_normalization(y_conv, training=is_training)

    return y_conv


def main(_):
    # Disable mnist dataset deprecation warning
    tf.logging.set_verbosity(tf.logging.ERROR)

    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    train_images = mnist.train.images
    mu_train = np.mean(train_images, axis=0)
    std_train = np.std(train_images, axis=0)
    std_train[std_train == 0] = 1

    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])

    train_phase = tf.placeholder(tf.bool, name="is_training")

    # Build the graph for the deep net
    y_conv = cryptonets_train(x, train_phase)

    print('tf.trainable_variables', tf.trainable_variables())
    print('tf.all_variables', tf.all_variables())

    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=y_, logits=y_conv)
    cross_entropy = tf.reduce_mean(cross_entropy)

    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        loss_values = []
        for i in range(FLAGS.train_loop_count):
            batch = mnist.train.next_batch(FLAGS.batch_size)
            x_train = (batch[0] - mu_train) / std_train
            if i % 100 == 0:
                t = time.time()
                train_accuracy = accuracy.eval(feed_dict={
                    x: x_train,
                    y_: batch[1],
                    train_phase: True
                })
                print('step %d, training accuracy %g, %g msec to evaluate' %
                      (i, train_accuracy, 1000 * (time.time() - t)))
            t = time.time()
            _, _, loss = sess.run([train_step, update_ops, cross_entropy],
                                  feed_dict={
                                      x: x_train,
                                      y_: batch[1],
                                      train_phase: True
                                  })
            loss_values.append(loss)

            if i % 1000 == 999 or i == FLAGS.train_loop_count - 1:
                x_test = mnist.test.images[:FLAGS.test_image_count]
                x_test = (x_test - mu_train) / std_train
                y_test = mnist.test.labels[:FLAGS.test_image_count]

                test_accuracy = accuracy.eval(feed_dict={
                    x: x_test,
                    y_: y_test,
                    train_phase: False
                })
                print('test accuracy %g' % test_accuracy)

        print("Training finished. Saving variables.")
        for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
            weight = (sess.run([var]))[0].flatten().tolist()
            filename = (str(var).split())[1].replace('/', '_')
            filename = filename.replace("'", "").replace(':0', '') + '.txt'

            if 'normalization' not in filename:
                weight = np.sign(weight)
            print("saving", filename)
            np.savetxt(str(filename), weight)


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
