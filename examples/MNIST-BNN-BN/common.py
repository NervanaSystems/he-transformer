import tensorflow as tf


def conv2d_stride_2_valid(x, W, name=None):
    """returns a 2d convolution layer with stride 2, valid pooling"""
    return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='VALID')


def avg_pool_3x3_same_size(x):
    """3x3 avg_pool using same padding, keeping original feature map size"""
    return tf.nn.avg_pool(
        x, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')


SCALING = 1
FC1_SIZE = int(845 * SCALING)
FC2_SIZE = int(100 * SCALING)
NUM_KERNELS = int(5 * SCALING)

print('NUM_KERNELS', NUM_KERNELS)
print('FC1_SIZE', FC1_SIZE)
print('FC2_SIZE', FC2_SIZE)