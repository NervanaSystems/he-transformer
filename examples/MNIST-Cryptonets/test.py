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
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    paddings = tf.constant([[0, 0], [0, 1], [0, 1], [0, 0]], name='pad_const')
    x_image = tf.pad(x_image, paddings)

    W_conv1 = common.get_variable('W_conv1', [5, 5, 1, 5], 'test')
    y = common.conv2d_stride_2_valid(x_image, W_conv1)
    y = tf.square(y)
    W_squash = common.get_variable('W_squash', [5 * 13 * 13, 100], 'test')
    y = tf.reshape(y, [-1, 5 * 13 * 13])
    y = tf.matmul(y, W_squash)
    y = tf.square(y)
    W_fc2 = common.get_variable('W_fc2', [100, 10], 'test')
    y = tf.matmul(y, W_fc2)
    return y


def test_mnist_cnn(FLAGS):

    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])

    y_conv = cryptonets_test_squashed(x)

    with tf.Session() as sess:
        x_test = mnist.test.images[:FLAGS.batch_size]
        y_test = mnist.test.labels[:FLAGS.batch_size]
        start_time = time.time()
        y_conv_val = y_conv.eval(feed_dict={x: x_test, y_: y_test})
        elasped_time = (time.time() - start_time)
        print("total time(s)", np.round(elasped_time, 3))

    x_test_batch = mnist.test.images[:FLAGS.batch_size]
    y_test_batch = mnist.test.labels[:FLAGS.batch_size]
    x_test = mnist.test.images
    y_test = mnist.test.labels

    y_label_batch = np.argmax(y_test_batch, 1)

    if FLAGS.save_batch:
        x_test_batch.tofile("x_test_" + str(FLAGS.batch_size) + ".bin")
        y_label_batch.astype('float32').tofile("y_label_" +
                                               str(FLAGS.batch_size) + ".bin")

    correct_prediction = np.equal(np.argmax(y_conv_val, 1), y_label_batch)
    error_count = np.size(correct_prediction) - np.sum(correct_prediction)
    test_accuracy = np.mean(correct_prediction)

    print('Error count:', error_count, 'of', FLAGS.batch_size, 'elements.')
    print('Accuracy: ', test_accuracy)

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

    # Test using squashed graph
    test_mnist_cnn(FLAGS)


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

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)