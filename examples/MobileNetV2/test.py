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
import json
import argparse
import os
import time
import PIL
from PIL import Image
import multiprocessing as mp


def print_nodes(filename):
    graph_def = load_model(filename)
    nodes = [n.name for n in graph_def.node]
    print('nodes', len(nodes))
    for node in sorted(nodes):
        print(node)


def load_model(filename):
    print("loading graph", filename)
    sess = tf.Session()
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
    grace_hopper = Image.open(grace_hopper).resize((FLAGS.image_size,
                                                    FLAGS.image_size))
    grace_hopper = np.array(grace_hopper) / 255.0
    print(grace_hopper.shape)

    grace_hopper = np.expand_dims(grace_hopper, axis=0)
    return grace_hopper


def get_imagenet_training_labels():
    labels_path = tf.keras.utils.get_file(
        'ImageNetLabels.txt',
        'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt'
    )
    imagenet_labels = np.array(open(labels_path).read().splitlines())
    return imagenet_labels


def get_imagenet_inference_labels():
    filename = "https://gist.githubusercontent.com/aaronpolhamus/964a4411c0906315deb9f4a3723aac57/raw/aa66dd9dbf6b56649fa3fab83659b2acbf3cbfd1/map_clsloc.txt"

    labels_path = tf.keras.utils.get_file('map_clsloc.txt', filename)
    labels = open(labels_path).read().splitlines()
    labels = ['background'] + [label.split()[2] for label in labels]
    labels = [label.replace('_', ' ') for label in labels]
    labels = np.array(labels)
    return labels


def get_validation_labels(FLAGS):
    data_dir = FLAGS.data_dir
    ground_truth_filename = 'ILSVRC2012_validation_ground_truth.txt'

    truth_file = os.path.join(data_dir, ground_truth_filename)
    if not os.path.isfile(truth_file):
        print('Cannot find ', ground_truth_filename, ' in ', data_dir)
        print('File ', truth_file, ' does not exist')
        exit(1)

    truth_labels = np.loadtxt(truth_file, dtype=np.int32)
    assert (truth_labels.shape == (50000, ))
    return truth_labels[0:FLAGS.batch_size]


def center_crop(im, new_size):
    # Center-crop image
    width, height = im.size
    left = (width - new_size) / 2
    top = (height - new_size) / 2
    right = (width + new_size) / 2
    bottom = (height + new_size) / 2
    im = im.crop((left, top, right, bottom))
    assert (im.size == (new_size, new_size))
    return im


def center_crop2(im, new_size):
    # Resize such that shortest side has new_size
    width, height = im.size
    ratio = min(width / new_size, height / new_size)
    im = im.resize((int(width / ratio), int(height / ratio)),
                   resample=Image.LANCZOS)

    # Center crop to new_size x new_size
    im = center_crop(im, new_size)
    return im


def get_validation_image(i):
    image_num_str = str(i + 1).zfill(8)
    data_dir = FLAGS.data_dir
    crop_size = FLAGS.crop_size

    image_prefix = 'validation_images/ILSVRC2012_val_' + image_num_str
    image_suffix = '.JPEG'
    image_name = image_prefix + image_suffix

    crop_filename = os.path.join(data_dir, image_prefix + '_crop' + '.png')

    filename = os.path.join(data_dir, image_name)
    if FLAGS.batch_size < 10:
        print('opening image at', filename)
    if not os.path.isfile(filename):
        print('Cannot find image ', filename)
        exit(1)

    if FLAGS.load_cropped_images:
        im = Image.open(crop_filename)
    else:
        im = Image.open(filename)
        im = center_crop2(im, crop_size)
        im = im.resize((FLAGS.image_size, FLAGS.image_size), PIL.Image.LANCZOS)

    # Fix grey images
    if im.mode != "RGB":
        im = im.convert(mode="RGB")
    assert (im.size == (FLAGS.image_size, FLAGS.image_size))

    if FLAGS.save_images:
        im.save(crop_filename, "PNG")
    im = np.array(im, dtype=np.float)

    assert (im.shape == (FLAGS.image_size, FLAGS.image_size, 3))

    # Standardize to [-1,1]
    if FLAGS.standardize:
        im = im / 255.
        means = [0.485, 0.456, 0.406]
        stds = [0.229, 0.224, 0.225]
        # Subtract mean, then scale such that result is in (-1, 1)
        for channel in range(3):
            im[:, :, channel] = (im[:, :, channel] - means[channel]) * (
                1. / means[channel])
    else:
        im = im / 128. - 1
    im = np.expand_dims(im, axis=0)
    return im


def get_validation_images(FLAGS, crop=False):
    print('getting validation images')
    images = np.empty((FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size,
                       3))
    pool = mp.Pool()
    images[:] = pool.map(get_validation_image, range(FLAGS.batch_size))
    pool.close()
    print('got validation images')
    return images


def accuracy(preds, truth):
    num_preds = truth.shape[0]

    if num_preds == 1:
        top1_cnt = int(truth[0] == preds[0])
        top5_cnt = int(truth[0] in preds)
    else:
        top1_cnt = 0
        top5_cnt = 0
        for i in range(num_preds):
            if preds[i][0] == truth[i]:
                top1_cnt += 1
            if truth[i] in preds[i]:
                top5_cnt += 1

    top5_acc = top5_cnt / float(num_preds) * 100.
    top1_acc = top1_cnt / float(num_preds) * 100.

    print('top1_acc', np.round(top1_acc, 3))
    print('top5_acc', np.round(top5_acc, 3))


def main(FLAGS):
    imagenet_inference_labels = get_imagenet_inference_labels()
    imagenet_training_labels = get_imagenet_training_labels()

    assert (
        sorted(imagenet_training_labels) == sorted(imagenet_inference_labels))

    validation_nums = get_validation_labels(FLAGS)
    x_test = get_validation_images(FLAGS)
    validation_labels = imagenet_inference_labels[validation_nums]

    if FLAGS.ngraph:
        import ngraph_bridge

    config = tf.ConfigProto()
    config.intra_op_parallelism_threads = 44
    config.inter_op_parallelism_threads = 44
    config = ngraph_bridge.update_config(config)
    sess = tf.Session(config=config)
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
    print('performed inference, runtime (s): ', np.round(runtime, 2))
    print('runtime per image (s)', np.round(per_image_runtime, 2))
    y_pred = np.squeeze(y_pred)
    # print(y_pred.shape)

    if (FLAGS.batch_size == 1):
        top5 = y_pred.argsort()[-5:]
    else:
        top5 = np.flip(y_pred.argsort()[:, -5:], axis=1)
    #print('top5', top5)

    preds = imagenet_training_labels[top5]

    if FLAGS.batch_size < 10:
        print('validation_labels', validation_labels)
        print('preds', preds)

    accuracy(preds, validation_labels)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


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