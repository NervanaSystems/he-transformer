import time
import argparse
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)

import keras
from keras.models import model_from_json
from keras import backend as K
# CPU needs NHWC format for MaxPool / FusedBatchNorm
keras.backend.set_image_data_format('channels_last')
from keras.callbacks import ModelCheckpoint
from keras.datasets import cifar10

import he_seal_client

FLAGS = None


def test_cifar10():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    print(x_test.shape)

    batch_size = FLAGS.batch_size
    # x_test_batch = mnist.test.images[:batch_size]

    x_test_batch = x_test[:FLAGS.batch_size]
    y_test_batch = x_test[:FLAGS.batch_size]

    print('x_test_batch', x_test_batch.shape)
    #print('x_test_batch', x_test_batch)

    data = x_test_batch.flatten('F')

    port = 34000
    client = he_seal_client.HESealClient(FLAGS.hostname, port, batch_size,
                                         data)

    print('Sleeping until client is done')
    while not client.is_done():
        time.sleep(1)

    results = client.get_results()
    results = np.round(results, 2)
    #print('results', results)

    y_pred = np.array(results).reshape(10, batch_size).argmax(axis=0)
    y_true = y_test_batch.argmax(axis=1)

    correct = np.sum(np.equal(y_pred, y_true))
    acc = correct / float(batch_size)
    print('correct', correct)
    print('Accuracy (batch size', batch_size, ') =', acc * 100., '%')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batch_size', type=int, default=50, help='Batch Size')
    parser.add_argument(
        '--hostname',
        type=str,
        default='localhost',
        help='Host where server is')
    FLAGS, unparsed = parser.parse_known_args()
    test_cifar10()
