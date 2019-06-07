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
import argparse
import os
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


def get_images(FLAGS):
    data_dir = FLAGS.data_dir
    ground_truth_filename = 'ILSVRC2012_validation_ground_truth.txt'

    truth_file = os.path.join(data_dir, ground_truth_filename)
    if not os.path.isfile(truth_file):
        print('Cannot find ', ground_truth_filename, ' in ', data_dir)
        print('File ', truth_file, ' does not exist')
        exit(1)

    truth_labels = np.loadtxt(truth_file, dtype=np.int32)
    print(truth_labels.shape)

    assert (truth_labels.shape == (50000, ))

    print('loaded truth_labels')


def main(FLAGS):

    get_images(FLAGS)

    exit(1)

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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        type=str,
        default=None,
        help=
        'Directory where cropped ImageNet data and ground truth labels are stored'
    )
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')

    FLAGS, unparsed = parser.parse_known_args()
    main(FLAGS)