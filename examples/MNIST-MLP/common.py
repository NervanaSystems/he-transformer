import tensorflow as tf
import numpy as np


def conv2d_stride_2_valid(x, W, name=None):
    """returns a 2d convolution layer with stride 2, valid pooling"""
    return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='VALID')


def avg_pool_3x3_same_size(x):
    """3x3 avg_pool using same padding, keeping original feature map size"""
    return tf.nn.avg_pool(
        x, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')


def mlp_model(x, mode):
    if mode not in set(['train', 'test']):
        print('mode should be train or test')
        raise Exception()

    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, 784])

    with tf.name_scope('fc1'):
        if mode == 'train':
            W_fc1 = tf.get_variable("W_fc1", [784, 10])
        else:
            W_fc1 = tf.constant(
                np.loadtxt('W_fc1.txt', dtype=np.float32).reshape([784, 10]))

        y_conv = tf.matmul(x_image, W_fc1)
    return y_conv