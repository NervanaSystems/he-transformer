import time
import argparse
import numpy as np
import sys
import os

np.set_printoptions(threshold=sys.maxsize)

from tensorflow.examples.tutorials.mnist import input_data
import he_seal_client

FLAGS = None


def test_mnist_cnn(FLAGS):
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    batch_size = FLAGS.batch_size
    x_test_batch = mnist.test.images[:batch_size]
    y_test_batch = mnist.test.labels[:batch_size]

    data = x_test_batch.flatten('F')
    print('Client batch size from FLAG: ', batch_size)

    complex_scale_factor = 1
    if ('NGRAPH_COMPLEX_PACK' in os.environ):
        complex_scale_factor = 2

    print('complex_scale_factor', complex_scale_factor)

    # TODO: support even batch sizes
    assert (batch_size % complex_scale_factor == 0)

    hostname = 'localhost'
    port = 34000

    new_batch_size = batch_size // complex_scale_factor
    print('new_batch_size', new_batch_size)

    client = he_seal_client.HESealClient(hostname, port, new_batch_size, data)

    print('Sleeping until client is done')
    while not client.is_done():
        time.sleep(1)

    results = client.get_results()
    results = np.round(results, 2)
    print('results', results)

    y_pred_reshape = np.array(results).reshape(10, batch_size)
    print('y_pred_reshape', y_pred_reshape)

    y_pred = y_pred_reshape.argmax(axis=0)
    print('y_pred', y_pred)
    y_true = y_test_batch.argmax(axis=1)

    correct = np.sum(np.equal(y_pred, y_true))
    acc = correct / float(batch_size)
    print('pred size', len(y_pred))
    print('correct', correct)
    print('Accuracy (batch size', batch_size, ') =', acc * 100., '%')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/tmp/tensorflow/mnist/input_data',
        help='Directory where input data is stored')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')

    FLAGS, unparsed = parser.parse_known_args()

    test_mnist_cnn(FLAGS)
