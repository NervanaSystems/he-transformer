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

import time
import argparse
import numpy as np
import sys
import os

from mnist_util import load_mnist_data, str2bool
import pyhe_client

FLAGS = None


def test_mnist_cnn(FLAGS):
    (x_train, y_train, x_test, y_test) = load_mnist_data()

    batch_size = FLAGS.batch_size
    x_test_batch = x_test[:batch_size]
    y_test_batch = y_test[:FLAGS.batch_size]

    data = x_test_batch.flatten('C')
    print('Client batch size from FLAG:', batch_size)

    complex_packing = False
    if ('NGRAPH_COMPLEX_PACK' in os.environ):
        complex_packing = str2bool(os.environ['NGRAPH_COMPLEX_PACK'])

    hostname = 'localhost'
    port = 34000

    print('complex_packing?', complex_packing)

    client = pyhe_client.HESealClient(hostname, port, batch_size, data,
                                      complex_packing)

    print('Sleeping until client is done')
    while not client.is_done():
        time.sleep(1)

    results = client.get_results()
    results = np.round(results, 2)

    y_pred_reshape = np.array(results).reshape(batch_size, 10)
    with np.printoptions(precision=3, suppress=True):
        print(y_pred_reshape)

    y_pred = y_pred_reshape.argmax(axis=1)
    print('y_pred', y_pred)
    y_true = y_test_batch.argmax(axis=1)

    correct = np.sum(np.equal(y_pred, y_true))
    acc = correct / float(batch_size)
    print('pred size', len(y_pred))
    print('correct', correct)
    print('Accuracy (batch size', batch_size, ') =', acc * 100., '%')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')

    FLAGS, unparsed = parser.parse_known_args()

    test_mnist_cnn(FLAGS)
