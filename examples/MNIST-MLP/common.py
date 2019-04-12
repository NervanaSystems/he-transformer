import tensorflow as tf
import numpy as np


def conv2d_stride_2_valid(x, W, name=None):
    """returns a 2d convolution layer with stride 2, valid pooling"""
    return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='VALID')


def avg_pool_3x3_same_size(x):
    """3x3 avg_pool using same padding, keeping original feature map size"""
    return tf.nn.avg_pool(
        x, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')


def get_variable(name, shape, mode):
    if mode not in set(['train', 'test']):
        print('mode should be train or test')
        raise Exception()

    if mode == 'train':
        return tf.get_variable(name, shape)
    else:
        return tf.constant(
            np.loadtxt(name + '.txt', dtype=np.float32).reshape(shape))


def mlp_model(x, mode):
    if mode not in set(['train', 'test']):
        print('mode should be train or test')
        raise Exception()

    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, 28, 28, 1])

    with tf.name_scope('conv1'):
        W_conv1 = get_variable('W_conv1', [5, 5, 1, 5], mode)
        y = conv2d_stride_2_valid(x_image, W_conv1)
        paddings = tf.constant([[0, 0], [0, 1], [0, 1], [0, 0]],
                               name='pad_const')
        y = tf.pad(y, paddings)
        y = tf.nn.relu(y)
        y = tf.math.minimum(y, 6)  # Use ReLU6 op
        y = tf.nn.max_pool(
            y, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.name_scope('fc1'):
        y = tf.reshape(y, [-1, 7 * 7 * 5])
        W_fc1 = get_variable('W_fc1', [7 * 7 * 5, 100], mode)
        y = tf.matmul(y, W_fc1)
        y = tf.nn.relu(y)

    with tf.name_scope('fc2'):
        W_fc2 = get_variable('W_fc2', [100, 10], mode)
        y = tf.matmul(y, W_fc2)
    return y
