import time
import argparse

from tensorflow.examples.tutorials.mnist import input_data
import he_seal_client

FLAGS = None


def test_mnist_cnn(FLAGS):
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    x_test_batch = mnist.test.images[:FLAGS.batch_size]
    y_test_batch = mnist.test.labels[:FLAGS.batch_size]

    print('x_test_batch', x_test_batch)

    print('x_test_batch', x_test_batch.shape)

    data = x_test_batch.flatten()

    print('data', data.shape)

    hostname = 'localhost'
    port = 34000

    client = he_seal_client.HESealClient(hostname, port, data)

    print('Sleeping until client is done')
    while not client.is_done():
        time.sleep(1)

    results = client.get_results()

    print('results', results)


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
