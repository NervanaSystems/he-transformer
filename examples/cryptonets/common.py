import tensorflow as tf


def conv2d(x, W, name=None):
    """conv2d returns a 2d convolution layer with stride 2."""
    return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME')


def scaled_mean_pool_2x2(x):
    """scaled_mean_pool_2x keeps feature map size."""
    return tf.nn.avg_pool(
        x, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
