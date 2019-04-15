import tensorflow as tf
from tensorflow.python.platform import gfile

import ngraph_bridge
import json
import he_seal_client
import time
import numpy as np

from PIL import Image


def print_nodes(filename):
    graph_def = read_pb_file(filename)
    nodes = [n.name for n in graph_def.node]
    print('nodes', len(nodes))
    for node in sorted(nodes):
        print(node)


def read_pb_file(filename):
    sess = tf.Session()
    print("load graph", filename)
    with gfile.GFile(filename, 'rb') as f:
        graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')

    return graph_def


def get_test_image():
    # https://www.tensorflow.org/tutorials/images/hub_with_keras
    # https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg
    filename = './images/grace_hopper.jpg'
    grace_hopper = Image.open(filename).resize((84, 84))
    grace_hopper = np.array(grace_hopper) / 255.0
    print(grace_hopper.shape)

    grace_hopper = np.expand_dims(grace_hopper, axis=0)
    return grace_hopper


def get_imagenet_labels():
    labels_path = tf.keras.utils.get_file(
        'ImageNetLabels.txt',
        'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt'
    )
    imagenet_labels = np.array(open(labels_path).read().splitlines())

    return imagenet_labels


def main():
    x_test = get_test_image()
    x_test = x_test.flatten('F')

    hostname = 'localhost'
    port = 34000
    batch_size = 1

    client = he_seal_client.HESealClient(hostname, port, batch_size, x_test)

    while not client.is_done():
        time.sleep(1)
    results = client.get_results()
    print('results', results)


if __name__ == '__main__':
    main()
