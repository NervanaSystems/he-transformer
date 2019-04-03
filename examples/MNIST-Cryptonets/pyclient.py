import time
import argparse
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
import he_seal_client

FLAGS = None


def test_mnist_cnn(FLAGS):
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    batch_size = FLAGS.batch_size

    x_test_batch = mnist.test.images[:batch_size]
    y_test_batch = mnist.test.labels[:batch_size]

    print('x_test_batch', x_test_batch)

    print('x_test_batch', x_test_batch.shape)

    data = x_test_batch.flatten('A')

    print('data', data.shape)

    print('batch size', batch_size)

    hostname = 'localhost'
    port = 34000

    client = he_seal_client.HESealClient(hostname, port, batch_size, data)

    print('Sleeping until client is done')
    while not client.is_done():
        time.sleep(1)

    results = client.get_results()
    print('results', results)

    results = np.array(results).reshape(10, batch_size)

    y_pred = results.argmax(axis=0)
    print('y_pred', y_pred)
    print('y_test_batch', y_test_batch)
    y_true = y_test_batch.argmax(axis=1)
    print('y_true', y_true)

    correct = np.sum(np.equal(y_pred, y_true))
    acc = correct / float(batch_size)
    print('correct', correct)
    print('Accuracy (batch size ', batch_size, ') =', acc * 100., '%')


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
