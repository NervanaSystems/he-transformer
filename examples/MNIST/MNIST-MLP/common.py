import tensorflow as tf
import numpy as np


def mlp_model(x, mode):
    if mode not in set(['train', 'test']):
        print('mode should be train or test')
        raise Exception()

    with tf.name_scope('conv1'):
        W_conv1 = get_variable('W_conv1', [5, 5, 1, 5], mode)
        y = conv2d_stride_2_valid(x_image, W_conv1)
        # paddings = tf.constant([[0, 0], [0, 1], [0, 1], [0, 0]],
        #                        name='pad_const')
        #y = tf.pad(y, paddings)
        y = tf.nn.relu(y)
        y = tf.math.minimum(y, 6)  # Use ReLU6 op
        y = tf.nn.max_pool(
            y, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.name_scope('fc1'):
        y = tf.reshape(y, [-1, 6 * 6 * 5])
        W_fc1 = get_variable('W_fc1', [6 * 6 * 5, 100], mode)
        y = tf.matmul(y, W_fc1)
        W_bias1 = get_variable('W_b1', [100], mode)
        W_scale1 = get_variable('W_s1', [100], mode)
        y = y * W_scale1
        y = tf.nn.relu(y)
        y = y + W_bias1
        y = tf.nn.relu(y)
        y = tf.math.minimum(y, 6)  # Use ReLU6 op

    with tf.name_scope('fc2'):
        W_fc2 = get_variable('W_fc2', [100, 10], mode)
        y = tf.matmul(y, W_fc2)
    return y
