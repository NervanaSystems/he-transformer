# ******************************************************************************
# Copyright 2018-2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# *****************************************************************************

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
    grace_hopper = tf.keras.utils.get_file(
        'grace_hopper.jpg',
        'https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg'
    )
    grace_hopper = Image.open(grace_hopper).resize((96, 96))
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
    # print(y_test)
    print(y_test.shape)

    top5 = y_test.argsort()[-5:][::-1]

    print('top5', top5)

    preds = imagenet_labels[top5]
    print(preds)


if __name__ == '__main__':
    main()