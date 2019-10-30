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
from tensorflow import keras
from tensorflow.python.platform import gfile
from tensorflow.python.keras.backend import set_session
import numpy as np
import json
import argparse
import os
import time
import PIL
from PIL import Image
import multiprocessing as mp
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import util
from util import get_imagenet_inference_labels, \
                 get_imagenet_training_labels, \
                 get_validation_image, \
                 get_validation_images, \
                 get_validation_labels, \
                 str2bool, \
                 server_config_from_flags, \
                 server_argument_parser

FLAGS = None


def load_model(filename):
    print("loading graph", filename)
    sess = tf.compat.v1.Session()
    with gfile.GFile(filename, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')
    return graph_def


def print_nodes(filename):
    graph_def = load_model(filename)
    nodes = [n.name for n in graph_def.node]
    print('nodes', len(nodes))
    for node in sorted(nodes):
        print(node)


def main(FLAGS):
    config = server_config_from_flags(FLAGS, 'input_1')
    sess = tf.compat.v1.Session(config=config)

    set_session(sess)

    model = keras.applications.densenet.DenseNet121(
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        input_shape=(FLAGS.image_size, FLAGS.image_size, 3),
        pooling='max',
        classes=1000)
    print('loaded keras model')
    #print(model.summary())

    imagenet_inference_labels = get_imagenet_inference_labels()
    imagenet_training_labels = get_imagenet_training_labels()

    util.VAL_IMAGE_FLAGS = FLAGS

    assert (
        sorted(imagenet_training_labels) == sorted(imagenet_inference_labels))

    validation_nums = get_validation_labels(FLAGS)
    x_test = get_validation_images(FLAGS)
    validation_labels = imagenet_inference_labels[validation_nums]

    if FLAGS.ngraph:
        import ngraph_bridge
        print(ngraph_bridge.__version__)

    y_pred = model.predict(x_test)
    print('y_pred', y_pred.shape)

    preds = keras.applications.densenet.decode_predictions(y_pred, top=5)
    preds = np.array([[pred[top][1] for top in range(5)] for pred in preds])

    print('preds', preds)
    util.accuracy(preds, validation_labels)

    return

    config = tf.compat.v1.ConfigProto()
    config.intra_op_parallelism_threads = 44
    config.inter_op_parallelism_threads = 44
    if FLAGS.ngraph:
        config = ngraph_bridge.update_config(config)
    sess = tf.compat.v1.Session(config=config)
    graph_def = load_model(FLAGS.model)
    tf.import_graph_def(graph_def, name='')

    input_tensor = sess.graph.get_tensor_by_name('input:0')
    output_tensor = sess.graph.get_tensor_by_name(
        'MobilenetV2/Logits/Conv2d_1c_1x1/BiasAdd:0')

    print('performing inference')
    start_time = time.time()
    y_pred = sess.run(output_tensor, {input_tensor: x_test})
    end_time = time.time()
    runtime = end_time - start_time
    per_image_runtime = runtime / float(FLAGS.batch_size)
    print('performed inference, runtime (s):', np.round(runtime, 2))
    print('runtime per image (s)', np.round(per_image_runtime, 2))
    y_pred = np.squeeze(y_pred)

    if (FLAGS.batch_size == 1):
        top5 = y_pred.argsort()[-5:]
    else:
        top5 = np.flip(y_pred.argsort()[:, -5:], axis=1)

    if not using_client:
        preds = imagenet_training_labels[top5]

        if FLAGS.batch_size < 10:
            print('validation_labels', validation_labels)
            print('validation_labels shape', validation_labels.shape)
            print('preds', preds)
            print('preds shape', preds.shape)

        util.accuracy(preds, validation_labels)


if __name__ == '__main__':
    parser = server_argument_parser()
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
        '--ngraph', type=str2bool, default=False, help='use ngraph backend')
    parser.add_argument(
        '--start_batch', type=int, default=0, help='Test data start index')

    FLAGS, unparsed = parser.parse_known_args()
    if FLAGS.data_dir == None:
        print('data_dir must be specified')
        exit(1)

    print(FLAGS)
    main(FLAGS)