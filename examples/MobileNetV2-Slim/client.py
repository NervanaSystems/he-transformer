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
    # Width x Height
    im = Image.open(filename).resize((84, 84))
    im = np.array(im) / 255.0
    # Add batch axis in front
    im = np.expand_dims(im, axis=0)
    return im


def get_imagenet_labels():
    labels_path = tf.keras.utils.get_file(
        'ImageNetLabels.txt',
        'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt'
    )
    imagenet_labels = np.array(open(labels_path).read().splitlines())
    return imagenet_labels


def main():
    x_test = get_test_image()
    # print('x_test', x_test)
    print('x_test[0][1]', x_test[0][1])
    print('x_test[0][1].shape', x_test[0][1].shape)

    (batch_size, width, height, channels) = x_test.shape
    print('batch_size', batch_size)
    print('width', width)
    print('height', height)
    print('channels', channels)

    # Reshape to expected layer
    # TODO: more efficient
    x_test_flat = []
    for width_idx in range(width):
        for height_idx in range(height):
            for channel_idx in range(channels):
                x_test_flat.append(
                    x_test[0][width_idx][height_idx][channel_idx])

    hostname = 'localhost'
    port = 34000
    batch_size = 1

    client = he_seal_client.HESealClient(hostname, port, batch_size,
                                         x_test_flat)

    while not client.is_done():
        time.sleep(1)
    results = client.get_results()
    print('results', results)

    imagenet_labels = get_imagenet_labels()

    correct_label = np.where(imagenet_labels == 'military uniform')  # 653
    results = np.array(results)

    top1000 = results.argsort()[-1000:][::-1]

    preds = imagenet_labels[top1000]
    print('top100', preds[0:100])
    print('index of military uniform', np.where(preds == 'military uniform'))


if __name__ == '__main__':
    main()