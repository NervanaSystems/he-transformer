# Adapted from CIFAR10 tensorflow tutorial
#
# Utility module to provide an async batch queue for CIFAR10 inputs
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import xrange
from six.moves import urllib

import tensorflow as tf
import os
import sys
import tarfile
import multiprocessing
import math
import numpy as np

IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32
IMAGE_DEPTH = 3

# As a rule of thumb, use twice as many input threads as the number of CPU
NUM_THREADS = multiprocessing.cpu_count() * 2

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir', './data',
                           """Directory containing data sets """)
tf.app.flags.DEFINE_boolean('data_aug', False,
                            """Whether to perform data augmentation or not""")
tf.app.flags.DEFINE_integer('batch_size', 128, """Size of each batch.""")

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'


def maybe_download_and_extract(data_dir):
    """Download and extract the tarball from Alex's website."""
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(data_dir, filename)
    if not os.path.exists(filepath):

        def _progress(count, block_size, total_size):
            sys.stdout.write(
                '\r>> Downloading %s %.1f%%' %
                (filename,
                 float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    extracted_dir_path = os.path.join(data_dir, 'cifar-10-batches-bin')
    if not os.path.exists(extracted_dir_path):
        tarfile.open(filepath, 'r:gz').extractall(data_dir)


def resize_image(image):
    return tf.image.resize_image_with_crop_or_pad(image, 24, 24)


def distort_image(image):
    # Crop each image to a random smaller image
    distorted_image = tf.random_crop(image, [24, 24, 3])
    # Flip left/right
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    # Adjust brightness (doesn't seem to work without temp var)
    distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
    # Adjust contrast
    return tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)


def get_raw_input_data(test_data, data_dir):
    """Raw CIFAR10 input data ops using the Reader ops.
    Args:
        test_data: bool, indicating if one should use the test or train set.
        data_dir: Path to the CIFAR-10 data directory.
    Returns:
        image: an op producing a 32x32x3 float32 image
        label: an op producing an int32 label
    """

    # Verify first that we have a valid data directory
    if not os.path.exists(data_dir):
        raise ValueError("Data directory %s doesn't exist" % data_dir)

    # Construct a list of input file names
    batches_dir = os.path.join(data_dir, 'cifar-10-batches-bin')
    if test_data:
        filenames = [os.path.join(batches_dir, 'test_batch.bin')]
    else:
        filenames = [
            os.path.join(batches_dir, 'data_batch_%d.bin' % ii)
            for ii in xrange(1, 6)
        ]

    # Make sure all input files actually exist
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    # Create a string input producer to cycle over file names
    filenames_queue = tf.train.string_input_producer(filenames)

    # CIFAR data samples are stored as contiguous labels and images
    label_size = 1
    image_size = IMAGE_DEPTH * IMAGE_HEIGHT * IMAGE_WIDTH

    # Instantiate a fixed length file reader
    reader = tf.FixedLengthRecordReader(label_size + image_size)

    # Read from files
    key, value = reader.read(filenames_queue)
    record_bytes = tf.decode_raw(value, tf.uint8)

    # Extract label and cast to int32
    label = tf.cast(tf.slice(record_bytes, [0], [label_size]), tf.int32)

    # Extract image and cast to float32
    image = tf.cast(
        tf.slice(record_bytes, [label_size], [image_size]), tf.float32)

    # Images are stored as D x H x W vectors, but we want H x W x D
    # So we need to convert to a matrix
    image = tf.reshape(image, (IMAGE_DEPTH, IMAGE_HEIGHT, IMAGE_WIDTH))
    # Transpose dimensions
    image = tf.transpose(image, (1, 2, 0))

    return (image, label)


def eval_inputs(test_data, data_dir, batch_size):
    """Construct input for CIFAR evaluation using the Reader ops.
    Args:
        test_data: bool, indicating if one should use the test or train set.
        data_dir: Path to the CIFAR-10 data directory.
        batch_size: Number of images per batch.
    Returns:
        images: Images. 4D tensor [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3].
        labels: Labels. 1D tensor [batch_size].
    """

    # Transpose dimensions
    raw_image, label = get_raw_input_data(test_data, data_dir)
    print("Get got raw inpout data")

    # If needed, perform data augmentation
    if tf.app.flags.FLAGS.data_aug:
        image = resize_image(raw_image)
    else:
        image = raw_image

    # Normalize image (substract mean and divide by variance)
    float_image = tf.image.per_image_standardization(image)

    print("Normalized eval inputs")

    # Create a queue to extract batch of samples
    images, labels = tf.train.batch([float_image, label],
                                    batch_size=batch_size,
                                    num_threads=NUM_THREADS,
                                    capacity=3 * batch_size)

    # Display the training images in the visualizer
    #tf.summary.image('images', images)

    return images, tf.reshape(labels, [batch_size])


def train_inputs(data_dir):
    """Construct input for CIFAR training.
    Note that batch_size is a placeholder whose default value is the one
    specified during training. It can however be specified differently at
    inference time by passing it explicitly in the feed dict when sess.run is
    called.
    Args:
        data_dir: Path to the CIFAR-10 data directory.
    Returns:
        images: Images. 4D tensor [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3].
        labels: Labels. 1D tensor [batch_size].
    """

    # Transpose dimensions
    raw_image, label = get_raw_input_data(False, data_dir)

    # If needed, perform data augmentation
    if tf.app.flags.FLAGS.data_aug:
        image = distort_image(raw_image)
    else:
        image = raw_image

    # Normalize image (substract mean and divide by variance)
    float_image = tf.image.per_image_standardization(image)

    # Create a queue to extract batch of samples
    batch_size_tensor = tf.placeholder_with_default(FLAGS.batch_size, shape=[])
    images, labels = tf.train.shuffle_batch(
        [float_image, label],
        batch_size=batch_size_tensor,
        num_threads=NUM_THREADS,
        capacity=20000 + 3 * FLAGS.batch_size,
        min_after_dequeue=20000)

    # Display the training images in the visualizer
    tf.summary.image('images', images)

    return images, tf.reshape(labels, [-1])


def numpy_eval_inputs(test_data, data_dir, batch_size):
    """Construct numpy input for CIFAR evaluation.
    Args:
        test_data: bool, indicating if one should use the test or train set.
        data_dir: Path to the CIFAR-10 data directory.
        batch_size: Number of images per batch.
    Returns:
        images: . 4D tensor [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3].
        labels: Labels. 1D tensor [batch_size].
    """
    assert (test_data == True)
    IMAGE_SIZE = 24 if FLAGS.data_aug else 32

    num_examples = 10000 if test_data else 50000

    data_filename = os.path.join(FLAGS.data_dir,
                                 'eval_data_' + str(IMAGE_SIZE) + '.npy')
    label_filename = os.path.join(FLAGS.data_dir,
                                  'eval_label_' + str(IMAGE_SIZE) + '.npy')
    print('data_filename', data_filename)
    print('label_filename', label_filename)

    if not os.path.isfile(data_filename) or not os.path.isfile(label_filename):

        eval_data = eval_inputs(True, FLAGS.data_dir, batch_size)

        np_eval_data_and_labels = []

        coord = tf.train.Coordinator()
        with tf.Session() as sess:
            try:
                threads = []
                for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                    threads.extend(
                        qr.create_threads(
                            sess, coord=coord, daemon=True, start=True))

                num_iter = int(math.ceil(num_examples / batch_size))
                true_count = 0  # Counts the number of correct predictions.
                total_sample_count = num_iter * batch_size
                step = 0
                while step < num_iter and not coord.should_stop():
                    data_in = sess.run([eval_data])
                    np_eval_data_and_labels.extend(data_in)
                    print('len(np_eval_data_and_labels)',
                          len(np_eval_data_and_labels))
                    step += 1

            except Exception as e:  # pylint: disable=broad-except
                coord.request_stop(e)

                coord.request_stop()
                coord.join(threads, stop_grace_period_secs=10)

        np_eval_data = [xy[0] for xy in np_eval_data_and_labels]
        np_eval_labels = [xy[1] for xy in np_eval_data_and_labels]

        np.save(data_filename, np.vstack(np_eval_data).flatten())
        np.save(label_filename, np.vstack(np_eval_labels).flatten())

    np_eval_data = np.load(data_filename)
    np_label_data = np.load(label_filename)
    np_eval_data = np_eval_data.reshape(num_examples, IMAGE_SIZE, IMAGE_SIZE,
                                        3)

    # Split data into batch_size chunks
    np_eval_data = np.split(
        np_eval_data, range(batch_size, np_eval_data.shape[0], batch_size))
    np_label_data = np.split(
        np_label_data, range(batch_size, np_label_data.shape[0], batch_size))

    # np_eval_data = [x.reshape([-1, IMAGE_SIZE, IMAGE_SIZE, 3]) for x in np_eval_data]

    return np_eval_data, np_label_data
