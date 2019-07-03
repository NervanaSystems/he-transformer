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


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


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


VAL_IMAGE_FLAGS = None


def get_validation_image(i):
    image_num_str = str(i + 1).zfill(8)
    data_dir = VAL_IMAGE_FLAGS.data_dir
    crop_size = VAL_IMAGE_FLAGS.crop_size

    image_prefix = 'validation_images/ILSVRC2012_val_' + image_num_str
    image_suffix = '.JPEG'
    image_name = image_prefix + image_suffix

    crop_filename = os.path.join(data_dir, image_prefix + '_crop' + '.png')

    filename = os.path.join(data_dir, image_name)
    if VAL_IMAGE_FLAGS.batch_size < 10:
        print('opening image at', filename)
    if not os.path.isfile(filename):
        print('Cannot find image ', filename)
        exit(1)

    if VAL_IMAGE_FLAGS.load_cropped_images:
        im = Image.open(crop_filename)
    else:
        im = Image.open(filename)
        im = center_crop2(im, crop_size)
        im = im.resize(
            (VAL_IMAGE_FLAGS.image_size, VAL_IMAGE_FLAGS.image_size),
            PIL.Image.LANCZOS)

    # Fix grey images
    if im.mode != "RGB":
        im = im.convert(mode="RGB")
    assert (im.size == (VAL_IMAGE_FLAGS.image_size,
                        VAL_IMAGE_FLAGS.image_size))

    if VAL_IMAGE_FLAGS.save_images:
        im.save(crop_filename, "PNG")
    im = np.array(im, dtype=np.float)

    assert (im.shape == (VAL_IMAGE_FLAGS.image_size,
                         VAL_IMAGE_FLAGS.image_size, 3))

    # Standardize to [-1,1]
    if VAL_IMAGE_FLAGS.standardize:
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
    print('FLAGS', FLAGS)
    VAL_IMAGE_FLAGS = FLAGS
    images = np.empty((FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size,
                       3))
    pool = mp.Pool()

    images[:] = pool.map(get_validation_image, range(FLAGS.batch_size))
    pool.close()
    print('got validation images')
    return images


def accuracy(preds, truth):
    truth = truth.flatten()
    num_preds = truth.size
    print('num_preds', num_preds)

    if (preds.shape[0] != num_preds):
        preds = preds.T
    assert (preds.shape[0] == num_preds)

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

    print('Accuracy on ', num_preds, ' predictions:')
    print('top1_acc', np.round(top1_acc, 3))
    print('top5_acc', np.round(top5_acc, 3))
