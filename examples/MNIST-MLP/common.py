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


PAD = True


def mlp_model(x, mode):
    if mode not in set(['train', 'test']):
        print('mode should be train or test')
        raise Exception()

    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, 28, 28, 1])

    with tf.name_scope('conv1'):
        W_conv1 = get_variable('W_conv1', [5, 5, 1, 5], mode)
        h_conv1 = conv2d_stride_2_valid(x_image, W_conv1)
        if PAD:
            paddings = tf.constant([[0, 0], [0, 1], [0, 1], [0, 0]],
                                   name='pad_const')
            h_conv1 = tf.pad(h_conv1, paddings)
            h_conv1 = tf.reshape(h_conv1, [-1, 13 * 13 * 5])
        else:
            h_conv1 = tf.reshape(h_conv1, [-1, 12 * 12 * 5])

    with tf.name_scope('fc1'):
        if PAD:
            W_fc1 = get_variable('W_fc1', [13 * 13 * 5, 10], mode)
        else:
            W_fc1 = get_variable('W_fc1', [12 * 12 * 5, 10], mode)
        y_conv = tf.matmul(h_conv1, W_fc1)
        #y_conv = tf.square(y_conv)
    return y_conv
