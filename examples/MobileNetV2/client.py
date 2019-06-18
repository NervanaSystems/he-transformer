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

import ngraph_bridge
import json
import he_seal_client
import time
import numpy as np

from PIL import Image
from test import get_test_image


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


def get_imagenet_labels():
    labels_path = tf.keras.utils.get_file(
        'ImageNetLabels.txt',
        'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt'
    )
    imagenet_labels = np.array(open(labels_path).read().splitlines())
    return imagenet_labels


def main():
    x_test = get_test_image()

    (batch_size, width, height, channels) = x_test.shape
    print('batch_size', batch_size)
    print('width', width)
    print('height', height)
    print('channels', channels)

    # Reshape to expected format
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
    results = np.array(results)
    top5 = results.argsort()[-5:][::-1]

    preds = imagenet_labels[top5]
    print('top5', preds)


if __name__ == '__main__':
    main()