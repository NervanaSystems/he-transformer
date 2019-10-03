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

import ngraph_bridge
import argparse

import numpy as np
import tensorflow as tf
from tensorflow.core.protobuf import rewriter_config_pb2


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def client_config_from_flags(FLAGS):
    rewriter_options = rewriter_config_pb2.RewriterConfig()
    rewriter_options.meta_optimizer_iterations = (
        rewriter_config_pb2.RewriterConfig.ONE)
    rewriter_options.min_graph_nodes = -1
    client_config = rewriter_options.custom_optimizers.add()
    client_config.name = "ngraph-optimizer"
    client_config.parameter_map["ngraph_backend"].s = FLAGS.backend.encode()
    client_config.parameter_map["device_id"].s = b''
    client_config.parameter_map[
        "encryption_parameters"].s = FLAGS.encryption_parameters.encode()
    client_config.parameter_map['enable_client'].s = (str(
        FLAGS.enable_client)).encode()
    if FLAGS.enable_client:
        client_config.parameter_map[b.name].s = b'client_input'

    config = tf.compat.v1.ConfigProto()
    config.MergeFrom(
        tf.compat.v1.ConfigProto(
            graph_options=tf.compat.v1.GraphOptions(
                rewrite_options=rewriter_options)))

    return config


def main(FLAGS):

    a = tf.constant(np.array([[1, 2, 3, 4]]), dtype=np.float32)
    b = tf.compat.v1.placeholder(
        tf.float32, shape=(1, 4), name='client_parameter_name')
    c = tf.compat.v1.placeholder(tf.float32, shape=(1, 4))
    f = (a + b) * a + c

    # Create config to load parameter b from client
    config = client_config_from_flags(FLAGS)
    print('config', config)

    with tf.compat.v1.Session(config=config) as sess:
        f_val = sess.run(f, feed_dict={b: np.ones((1, 4)), c: np.ones((1, 4))})
        print("Result: ", f_val)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument(
        '--enable_client',
        type=str2bool,
        default=False,
        help='Enable the client')
    parser.add_argument(
        '--backend',
        type=str,
        default='HE_SEAL',
        help='Name of backend to use')
    parser.add_argument(
        '--encryption_parameters',
        type=str,
        default='',
        help=
        'Filename containing json description of encryption parameters, or json description itself'
    )

    FLAGS, unparsed = parser.parse_known_args()
    main(FLAGS)
