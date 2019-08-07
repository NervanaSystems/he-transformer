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
import os

from PIL import Image
from util import get_imagenet_inference_labels, \
                 get_imagenet_training_labels, \
                 get_validation_image, \
                 get_validation_images, \
                 get_validation_labels, \
                 str2bool
import util
import numpy as np

FLAGS = None


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
    util.VAL_IMAGE_FLAGS = FLAGS

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

    for image in (0, 1):
        pixels = [(0, 0, 0), (0, 0, 1), (0, 0, 2)]
        pixels += [(0, 1, 0), (0, 1, 1), (0, 2, 1)]

        for pixel in pixels:
            p1, p2, p3 = pixel
            print('image', image, 'pixel', pixel, x_test[image][p1][p2][p3])

    #exit(1)
    #print(x_test.shape)

    #x_test[1, :, :, :] = x_test[0, :, :, :]

    # TODO: more efficient
    print('flattening x_test')
    x_test_flat = []
    for image_idx in range(batch_size):
        for width_idx in range(width):
            for height_idx in range(height):
                for channel_idx in range(channels):
                    # TODO: use image_idx
                    x_test_flat.append(
                        x_test[image_idx][width_idx][height_idx][channel_idx])
    print('done flattening x_test')

    # Reshape to expected format (batch axes innermost)
    #x_test = np.moveaxis(x_test, 0, -1)
    #x_test_flat = x_test.flatten(order='C')
    hostname = 'localhost'
    port = 34000

    #for i in range(10):
    #    print(i, x_test_flat[i])

    #exit(1)

    if 'NGRAPH_COMPLEX_PACK' in os.environ:
        complex_packing = str2bool(os.environ['NGRAPH_COMPLEX_PACK'])
    else:
        complex_packing = False

    client = he_seal_client.HESealClient(FLAGS.hostname, port, batch_size,
                                         x_test_flat, complex_packing)

    while not client.is_done():
        time.sleep(1)
    results = client.get_results()

    imagenet_labels = get_imagenet_labels()
    results = np.array(results)

    if (FLAGS.batch_size == 1):
        top5 = results.argsort()[-5:]
    else:
        print('results shape', results.shape)
        results = np.reshape(results, (
            1001,
            FLAGS.batch_size,
        ))
        print('results.shape', results.shape)

        try:
            res_sort = results.argsort(axis=0)
            res_top5 = res_sort[-5:, :]
            top5x = np.flip(res_top5, axis=0)
            top5 = top5x
            top5 = top5.T
            print('top5.shape', top5.shape)
        except e:
            print('e', e)
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