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
from keras.models import Model
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
    (x_train, y_train, x_test, y_test) = load_mnist_data()

    # Set ngraph-bridge configuration
    config = server_config_from_flags(FLAGS, 'input')
    config = ngraph_bridge.update_config(config)
    sess = tf.compat.v1.Session(config=config)
    set_session(sess)

    # Load saved model
    model = tf.keras.models.load_model('model.h5')
    model.summary()

    y_pred = model.predict(x_test[0:FLAGS.batch_size])
    print('Test pred:', y_pred)

    y_test_batch = y_test[0:FLAGS.batch_size]
    y_label_batch = np.argmax(y_test_batch, 1)

    correct_prediction = np.equal(np.argmax(y_pred, 1), y_label_batch)
    error_count = np.size(correct_prediction) - np.sum(correct_prediction)
    test_accuracy = np.mean(correct_prediction)

    print('Error count:', error_count, 'of', FLAGS.batch_size, 'elements.')
    print('Accuracy: ', test_accuracy)


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
