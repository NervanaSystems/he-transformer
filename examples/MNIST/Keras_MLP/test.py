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
"""An MNIST classifier based on Cryptonets using convolutional layers. """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
from keras import losses
from keras.backend.tensorflow_backend import set_session

import ngraph_bridge
import argparse
import sys
import time
import numpy as np
import itertools
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mnist_util import load_mnist_data, \
                       server_config_from_flags, \
                       server_argument_parser


def test_mnist_mlp(FLAGS):
    config = server_config_from_flags(FLAGS, '')
    config = ngraph_bridge.update_config(config)
    sess = tf.Session(config=config)
    set_session(sess)

    model = tf.keras.models.load_model('model.h5')
    model.summary()

    # Remove activation layer
    model.pop()
    model.summary()

    (x_train, y_train, x_test, y_test) = load_mnist_data()

    y_pred = model.predict(x_test[0:FLAGS.batch_size])
    print('Test pred:', y_pred)


if __name__ == '__main__':
    parser = server_argument_parser()
    FLAGS, unparsed = parser.parse_known_args()

    if unparsed:
        print('Unparsed flags:', unparsed)
    if FLAGS.encrypt_server_data and FLAGS.enable_client:
        raise Exception(
            "encrypt_server_data flag only valid when client is not enabled. Note: the client can specify whether or not to encrypt the data using 'encrypt' or 'plain' in the configuration map"
        )

    test_mnist_mlp(FLAGS)
