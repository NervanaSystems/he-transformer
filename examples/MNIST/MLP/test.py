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
from mnist_util import load_mnist_data, get_variable, conv2d_stride_2_valid

FLAGS = None


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
            "import/Placeholder:0")

        # Create configuration to encrypt data
        rewriter_options = rewriter_config_pb2.RewriterConfig()
        rewriter_options.meta_optimizer_iterations = (
            rewriter_config_pb2.RewriterConfig.ONE)
        rewriter_options.min_graph_nodes = -1
        ngraph_optimizer = rewriter_options.custom_optimizers.add()
        ngraph_optimizer.name = "ngraph-optimizer"
        ngraph_optimizer.parameter_map["ngraph_backend"].s = b'HE_SEAL'
        ngraph_optimizer.parameter_map["device_id"].s = b''
        ngraph_optimizer.parameter_map[str(x_input)].s = b'encrypt'
        ngraph_optimizer.parameter_map['enable_client'].s = (str(
            FLAGS.enable_client)).encode()
        ngraph_optimizer.parameter_map['complex_packing'].s = (str(
            FLAGS.complex_packing)).encode()

        config = tf.compat.v1.ConfigProto()
        config.MergeFrom(
            tf.compat.v1.ConfigProto(
                graph_options=tf.compat.v1.GraphOptions(
                    rewrite_options=rewriter_options)))

        print('config', config)

        with tf.compat.v1.Session(config=config) as sess:
            start_time = time.time()
            y_conv_val = y_conv.eval(
                session=sess, feed_dict={
                    x_input: x_test,
                })
            elasped_time = (time.time() - start_time)
            print("total time(s)", np.round(elasped_time, 3))

    using_client = (os.environ.get('NGRAPH_ENABLE_CLIENT') is not None)

    if not using_client:
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument(
        '--enable_client', type=bool, default=False, help='Enable the client')
    parser.add_argument(
        '--complex_packing',
        type=bool,
        default=False,
        help='Whether or not to use complex packing')

    FLAGS, unparsed = parser.parse_known_args()
    test_mnist_mlp(FLAGS)
