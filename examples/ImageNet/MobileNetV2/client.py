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
import pyhe_client

import numpy as np
import argparse
import os
import util
import numpy as np
from PIL import Image
from util import get_imagenet_inference_labels, \
                 get_imagenet_training_labels, \
                 get_validation_image, \
                 get_validation_images, \
                 get_validation_labels, \
                 str2bool


def print_nodes(filename):
    graph_def = read_pb_file(filename)
    nodes = [n.name for n in graph_def.node]
    print('nodes', len(nodes))
    for node in sorted(nodes):
        print(node)


def read_pb_file(filename):
    sess = tf.compat.v1.Session()
    print("load graph", filename)
    with gfile.GFile(filename, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
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


def main(FLAGS):
    imagenet_inference_labels = get_imagenet_inference_labels()
    imagenet_training_labels = get_imagenet_training_labels()
    assert (
        sorted(imagenet_training_labels) == sorted(imagenet_inference_labels))
    validation_nums = get_validation_labels(FLAGS)
    x_test = get_validation_images(FLAGS)
    validation_labels = imagenet_inference_labels[validation_nums]

    if FLAGS.batch_size < 10:
        print('validation_labels', validation_labels)

    (batch_size, width, height, channels) = x_test.shape
    print('batch_size', batch_size)
    print('width', width)
    print('height', height)
    print('channels', channels)

    x_test_flat = x_test.flatten(order='C')
    port = 34000

    client = pyhe_client.HESealClient(FLAGS.hostname, port, batch_size,
                                      {'input': ('encrypt', x_test_flat)})

    results = client.get_results()

    imagenet_labels = get_imagenet_labels()
    results = np.array(results)

    if (FLAGS.batch_size == 1):
        top5 = results.argsort()[-5:]
    else:
        results = np.reshape(results, (FLAGS.batch_size, 1001))
        top5 = np.flip(results.argsort()[:, -5:], axis=1)

    preds = imagenet_labels[top5]
    print('validation_labels', validation_labels)
    print('top5', preds)

    util.accuracy(preds, validation_labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        type=str,
        default=None,
        help=
        'Directory where cropped ImageNet data and ground truth labels are stored'
    )
    parser.add_argument(
        '--image_size', type=int, default=96, help='image size')
    parser.add_argument(
        '--save_images',
        type=str2bool,
        default=False,
        help='save cropped images')
    parser.add_argument(
        '--load_cropped_images',
        type=str2bool,
        default=False,
        help='load saved cropped images')
    parser.add_argument(
        '--standardize',
        type=str2bool,
        default=False,
        help='subtract training set mean from each image')
    parser.add_argument(
        '--crop_size',
        type=int,
        default=256,
        help='crop to this size before resizing to image_size')
    parser.add_argument(
        '--hostname', type=str, default='localhost', help='server hostname')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument(
        '--start_batch', type=int, default=0, help='Test data start index')

    FLAGS, unparsed = parser.parse_known_args()
    if FLAGS.data_dir == None:
        print('data_dir must be specified')
        exit(1)

    print(FLAGS)
    main(FLAGS)