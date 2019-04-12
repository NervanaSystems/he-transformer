import tensorflow as tf
from tensorflow.python.platform import gfile

import numpy as np

import ngraph_bridge
import json

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
    grace_hopper = Image.open(filename).resize((128, 128))
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
    sess = tf.Session()
    graph_def = read_pb_file('./model/opt1.pb')
    imagenet_labels = get_imagenet_labels()

    #print_nodes('./model/opt1.pb')

    tf.import_graph_def(graph_def, name='')

    input_tensor = sess.graph.get_tensor_by_name('input:0')
    output_tensor = sess.graph.get_tensor_by_name(
        'MobilenetV2/Logits/Conv2d_1c_1x1/BiasAdd:0')

    print('input_tensor', input_tensor)
    print('output_tensor', output_tensor)

    x_test = get_test_image()
    print(x_test.shape)

    y_test = sess.run(output_tensor, {input_tensor: x_test})
    y_test = np.squeeze(y_test)
    print(y_test.shape)

    top3 = y_test.argsort()[-3:][::-1]

    print('top3', top3)

    preds = imagenet_labels[top3]
    print(preds)


if __name__ == '__main__':
    main()
