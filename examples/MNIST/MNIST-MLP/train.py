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


def main(_):
    # Disable mnist dataset deprecation warning
    tf.logging.set_verbosity(tf.logging.ERROR)

    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])

    # Build the graph for the deep net
    y_conv = common.mlp_model(x, 'train')

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

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        loss_values = []
        for i in range(FLAGS.train_loop_count):
            batch = mnist.train.next_batch(FLAGS.batch_size)
            if i % 100 == 0:
                t = time.time()
                train_accuracy = accuracy.eval(feed_dict={
                    x: batch[0],
                    y_: batch[1]
                })
                print('step %d, training accuracy %g, %g msec to evaluate' %
                      (i, train_accuracy, 1000 * (time.time() - t)))
            t = time.time()
            _, loss = sess.run([train_step, cross_entropy],
                               feed_dict={
                                   x: batch[0],
                                   y_: batch[1]
                               })
            loss_values.append(loss)

            if i % 1000 == 999 or i == FLAGS.train_loop_count - 1:
                x_test = mnist.test.images[:FLAGS.test_image_count]
                y_test = mnist.test.labels[:FLAGS.test_image_count]

                test_accuracy = accuracy.eval(feed_dict={
                    x: x_test,
                    y_: y_test
                })
                print('test accuracy %g' % test_accuracy)

        print("Training finished. Saving variables.")
        for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
            weight = (sess.run([var]))[0].flatten().tolist()
            filename = (str(var).split())[1].replace('/', '_')
            filename = filename.replace("'", "").replace(':0', '') + '.txt'

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
