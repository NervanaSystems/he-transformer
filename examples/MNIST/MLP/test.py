# ==============================================================================
#  Copyright 2018-2019 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ==============================================================================
"""An MNIST classifier using convolutional layers and relu activations. """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import time
import numpy as np
import itertools
import glob
import tensorflow as tf
import model
import ngraph_bridge
import os
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow.core.protobuf import rewriter_config_pb2

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mnist_util import load_mnist_data, \
                       get_variable, \
                       conv2d_stride_2_valid, \
                       str2bool, \
                       server_argument_parser, \
                       server_config_from_flags


def load_pb_file(filename):
    with tf.io.gfile.GFile(filename, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    print('Model restored')
    return graph_def


def test_mnist_mlp(FLAGS):
    (x_train, y_train, x_test, y_test) = load_mnist_data()
    x_test = x_test[:FLAGS.batch_size]
    y_test = y_test[:FLAGS.batch_size]

    graph_def = load_pb_file('./model/model.pb')

    with tf.Graph().as_default():
        tf.import_graph_def(graph_def)
        y_conv = tf.compat.v1.get_default_graph().get_tensor_by_name(
            "import/output:0")
        x_input = tf.compat.v1.get_default_graph().get_tensor_by_name(
            "import/input:0")

        config = server_config_from_flags(FLAGS, x_input.name)

        print('config', config)

        with tf.compat.v1.Session(config=config) as sess:
            start_time = time.time()
            y_conv_val = y_conv.eval(
                session=sess, feed_dict={
                    x_input: x_test,
                })
            elasped_time = (time.time() - start_time)
            print("total time(s)", np.round(elasped_time, 3))

    if not FLAGS.enable_client:
        y_test_batch = y_test[:FLAGS.batch_size]
        y_label_batch = np.argmax(y_test_batch, 1)

        y_pred = np.argmax(y_conv_val, 1)
        print('y_pred', y_pred)
        correct_prediction = np.equal(y_pred, y_label_batch)
        error_count = np.size(correct_prediction) - np.sum(correct_prediction)
        test_accuracy = np.mean(correct_prediction)

        print('Error count', error_count, 'of', FLAGS.batch_size, 'elements.')
        print('Accuracy: %g ' % test_accuracy)


if __name__ == '__main__':
    parser = server_argument_parser()
    FLAGS, unparsed = parser.parse_known_args()

    if unparsed:
        print('Unparsed flags:', unparsed)
    if FLAGS.encrypt_data and FLAGS.enable_client:
        raise Exception(
            "encrypt_data flag only valid when client is not enabled. Note: the client can specify whether or not to encrypt the data using 'encrypt' or 'plain' in the configuration map"
        )

    test_mnist_mlp(FLAGS)
