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
import argparse

from PIL import Image
from util import get_imagenet_inference_labels, \
                 get_imagenet_training_labels, \
                 get_validation_image, \
                 get_validation_images, \
                 get_validation_labels, \
                 str2bool
import util

FLAGS = None


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


def main(FLAGS):
    #x_test = get_test_image
    util.VAL_IMAGE_FLAGS = FLAGS

    imagenet_inference_labels = get_imagenet_inference_labels()
    imagenet_training_labels = get_imagenet_training_labels()
    assert (
        sorted(imagenet_training_labels) == sorted(imagenet_inference_labels))
    validation_nums = get_validation_labels(FLAGS)
    x_test = get_validation_images(FLAGS)
    validation_labels = imagenet_inference_labels[validation_nums]
    print('validation_labels', validation_labels)

    (batch_size, width, height, channels) = x_test.shape
    print('batch_size', batch_size)
    print('width', width)
    print('height', height)
    print('channels', channels)

    # Reshape to expected format
    # TODO: more efficient
    print('flattening x_test')
    x_test_flat = []
    for image_idx in range(batch_size):
        for width_idx in range(width):
            for height_idx in range(height):
                for channel_idx in range(channels):
                    x_test_flat.append(
                        x_test[image_idx][width_idx][height_idx][channel_idx])
    print('done flattening x_test')

    hostname = 'localhost'
    port = 34000
    # batch_size = 1

    client = he_seal_client.HESealClient(hostname, port, batch_size,
                                         x_test_flat)

    while not client.is_done():
        time.sleep(1)
    results = client.get_results()

    imagenet_labels = get_imagenet_labels()
    results = np.array(results)

    if (FLAGS.batch_size == 1):
        top5 = results.argsort()[-5:]
    else:
        top5 = np.flip(results.argsort()[:, -5:], axis=1)

    preds = imagenet_labels[top5]
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
        '--model',
        type=str,
        default='./model/mobilenet_v2_0.35_96_opt.pb',
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
        '--ngraph', type=str2bool, default=False, help='use ngraph backend')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')

    FLAGS, unparsed = parser.parse_known_args()
    if FLAGS.data_dir == None:
        print('data_dir must be specified')
        exit(1)

    print(FLAGS)
    main(FLAGS)