"""Cryptonets MNIST classifier"""
import ngraph_bridge
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

batch_size = 4096

mnist = input_data.read_data_sets(
    '/tmp/tensorflow/mnist/input_data', one_hot=True)

# Create inference network
parameter_0 = tf.placeholder(tf.float32, [None, 784])
reshape_5_7 = tf.reshape(parameter_0, [-1, 28, 28, 1])
constant_4 = tf.constant(
    np.loadtxt('W_conv1.txt', dtype='f').reshape([5, 5, 1, 5]))
convolution_8 = tf.nn.conv2d(
    reshape_5_7, constant_4, strides=[1, 2, 2, 1], padding='VALID')
multiply_10 = tf.square(convolution_8)
constant_3 = tf.constant([[0, 0], [0, 1], [0, 1], [0, 0]])
pad_11 = tf.pad(multiply_10, constant_3)
reshape_12 = tf.reshape(pad_11, [-1, 845])
constant_2 = tf.constant(
    np.loadtxt("W_squash.txt", dtype='f').reshape([845, 100]))
dot_13 = tf.matmul(reshape_12, constant_2)
multiply_14 = tf.square(dot_13)
constant_1 = tf.constant(np.loadtxt('W_fc2.txt', dtype='f').reshape([100, 10]))
y_conv = tf.matmul(multiply_14, constant_1)

# Run network
with tf.Session() as sess:
    x_test = mnist.test.images[:batch_size]
    y_conv_val = y_conv.eval(feed_dict={parameter_0: x_test})

# Compute accuracy
y_test = mnist.test.labels[:batch_size]
correct_prediction = np.equal(np.argmax(y_conv_val, 1), np.argmax(y_test, 1))
error_count = batch_size - np.sum(correct_prediction)
test_accuracy = np.mean(correct_prediction)
print('Error count', error_count, 'of', batch_size, 'elements. Accuracy:',
      test_accuracy)
