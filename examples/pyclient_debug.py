import time
import argparse
import numpy as np
import sys
import tensorflow as tf
import os

np.set_printoptions(threshold=sys.maxsize)

from tensorflow.examples.tutorials.mnist import input_data
import he_seal_client

FLAGS = None

batch_size = 2


def debug_server():
    x = tf.placeholder(tf.float32, [None, 2])

    b = np.array(range(batch_size)).reshape((-1, 2))
    w = np.ones((batch_size, 2))

    y = w * x + b * w

    with tf.Session() as sess:

        x_input = np.array([1, 2, 3, 4]).reshape((-1, 2))
        y_eval = y.eval(feed_dict={x: x_input})
        print('result', np.round(y_eval, 2))


def debug_client():
    hostname = 'localhost'
    port = 34000

    new_batch_size = batch_size // 2
    print('new_batch_size', new_batch_size)

    client = he_seal_client.HESealClient(hostname, port, new_batch_size, data)

    print('Sleeping until client is done')
    while not client.is_done():
        time.sleep(1)

    results = client.get_results()
    results = np.round(results, 2)

    with np.printoptions(precision=3, suppress=True):
        print('results', results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--role', type=str, help='client or server')

    FLAGS, unparsed = parser.parse_known_args()

    if FLAGS.role == 'server':
        debug_server()
    elif FLAGS.role == 'client':
        debug_client()