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
import tensorflow as tf
import model
import os
from tensorflow.python.tools import freeze_graph

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mnist_util import load_mnist_data, \
    get_variable, \
    conv2d_stride_2_valid, \
    avg_pool_3x3_same_size, \
    get_train_batch

FLAGS = None


def save_model(sess, directory, filename):
    if not os.path.exists(directory):
        os.makedirs(directory)

    saver = tf.compat.v1.train.Saver()
    ckpt_filepath = os.path.join(directory, filename + '.ckpt')
    saver.save(sess, ckpt_filepath)

    pbtxt_filename = filename + '.pbtxt'
    pbtxt_filepath = os.path.join(directory, pbtxt_filename)
    pb_filepath = os.path.join(directory, filename + '.pb')

    tf.io.write_graph(
        graph_or_graph_def=sess.graph_def,
        logdir=directory,
        name=pbtxt_filename,
        as_text=True)

    # Freeze graph to turn variables into constants
    freeze_graph.freeze_graph(
        input_graph=pbtxt_filepath,
        input_saver='',
        input_binary=False,
        input_checkpoint=ckpt_filepath,
        output_node_names='output',
        restore_op_name='save/restore_all',
        filename_tensor_name='save/Const:0',
        output_graph=pb_filepath,
        clear_devices=True,
        initializer_nodes='')

    print("Model saved to: %s" % pb_filepath)


def main(_):
    (x_train, y_train, x_test, y_test) = load_mnist_data()

    x = tf.compat.v1.placeholder(tf.float32, [None, 28, 28, 1])
    y_ = tf.compat.v1.placeholder(tf.float32, [None, 10])
    y_conv = model.mnist_mlp_model(x)

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
            if i % 1000 == 999 or i == FLAGS.train_loop_count - 1:
                test_accuracy = accuracy.eval(feed_dict={
                    x: x_test,
                    y_: y_test
                })
                print('test accuracy %g' % test_accuracy)

        print("Training finished. Saving model.")

        save_model(sess, './model', 'model')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train_loop_count',
        type=int,
        default=20000,
        help='Number of training iterations')
    parser.add_argument(
        '--batch_size', type=int, default=50, help='Batch Size')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
