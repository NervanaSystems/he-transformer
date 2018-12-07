# Python 2 & 3 compatibility
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

WEIGHT_DECAY = 1e2

class Model(object):

    def __init__(self, wd=WEIGHT_DECAY, dropout=0.0):

        self.wd = wd
        self.dropout = dropout
        self.sizes = []
        self.flops = []
        self.training = tf.placeholder_with_default(False, shape=[], name="training")

    def name_to_filename(name):
      """Given a variable name, returns the filename where to store it"""
      name = name.replace('/','_')
      name = name.replace(':0', '') + '.txt'
      return 'weights/' + name

    def _get_weights_var(self, name, shape, decay=False):
        """Helper to create an initialized Variable with weight decay.

        The Variable is initialized using a normal distribution whose variance
        is provided by the xavier formula (ie inversely proportional to the number
        of inputs)

        Args:
            name: name of the tensor variable
            shape: the tensor shape
            decay: a boolean indicating if we apply decay to the tensor weights
            using a regularization loss
            restore_saved: a boolean, true if we should read saved weights

        Returns:
            Variable Tensor
        """
        if self.training:
          # Declare an initializer for this variable
          initializer = tf.contrib.layers.xavier_initializer(uniform=False,dtype=tf.float32)
          # Declare variable (it is trainable by default)
          var = tf.get_variable(name=name,
                                shape=shape,
                                initializer=initializer,
                                dtype=tf.float32)
          if decay:
              # We apply a weight decay to this tensor var that is equal to the
              # model weight decay divided by the tensor size
              weight_decay = self.wd
              for x in shape:
                  weight_decay /= x
              # Weight loss is L2 loss multiplied by weight decay
              weight_loss = tf.multiply(tf.nn.l2_loss(var),
                                        weight_decay,
                                        name='weight_loss')
              # Add weight loss for this variable to the global losses collection
              tf.add_to_collection('losses', weight_loss)

          return var
        else:
          restore_filename = name_to_filename(name)
          print('restoring variable: ', restore_filename)
          return tf.constant(np.loadtxt(restore_filename, dtype=np.float32).reshape(shape))

    def conv_layer(self, inputs, size, filters, stride, decay, name, bn=False):
        channels = inputs.get_shape()[3]
        shape = [size, size, channels, filters]
        with tf.variable_scope(name + '/conv') as scope:
            weights = self._get_weights_var('weights',
                                            shape=shape,
                                            decay=decay)

            biases = tf.get_variable('biases',
                                    shape=[filters],
                                    dtype=tf.float32,
                                    initializer=tf.constant_initializer(0.0))
            conv = tf.nn.conv2d(inputs,
                                weights,
                                strides=[1,stride,stride,1],
                                padding='SAME')
            if bn:
                conv = tf.layers.batch_normalization(conv,
                                                     training=self.training)
            pre_activation = tf.nn.bias_add(conv, biases)

            outputs= tf.nn.relu(pre_activation, name=scope.name)

        # Evaluate layer size
        self.sizes.append((name,(1+size*size*int(channels))*filters))

        # Evaluate number of operations
        N, w, h, c = outputs.get_shape()
        # Number of convolutions
        num_flops = (1+2*int(channels)*size*size)*filters*int(w)*int(h)
        # Number of ReLU
        num_flops += 2*filters*int(w)*int(h)
        self.flops.append((name, num_flops))

        return outputs

    def pool_layer(self, inputs, size, stride, name):
        with tf.variable_scope(name) as scope:
            outputs = tf.nn.max_pool(inputs,
                                     ksize=[1,size,size,1],
                                     strides=[1,stride,stride,1],
                                     padding='SAME',
                                     name=name)

        return outputs

    def fc_layer(self, inputs, neurons, decay, name, relu=True, bn=False):
        with tf.variable_scope(name) as scope:
            if len(inputs.get_shape().as_list()) > 2:
                # We need to reshape inputs:
                #   [ batch size , w, h, c ] -> [ batch size, w x h x c ]
                # Batch size is a dynamic value, but w, h and c are static and
                # can be used to specifiy the reshape operation
                dim = np.prod(inputs.get_shape().as_list()[1:])
                reshaped = tf.reshape(inputs, shape=[-1, dim], name='reshaped')
            else:
                # No need to reshape inputs
                reshaped = inputs
            dim = reshaped.get_shape().as_list()[1]
            weights = self._get_weights_var('weights',
                                            shape=[dim,neurons],
                                            decay=decay)
            biases = tf.get_variable('biases',
                                    shape=[neurons],
                                    dtype=tf.float32,
                                    initializer=tf.constant_initializer(0.0))
            x = tf.add(tf.matmul(reshaped, weights), biases)
            if bn:
                x = tf.layers.batch_normalization(x, training=self.training)
            if relu:
                outputs = tf.nn.relu(x)
            else:
                outputs = x

        # Evaluate layer size
        self.sizes.append((name, (dim + 1) * neurons))

        # Evaluate layer operations
        # Matrix multiplication plus bias
        num_flops = (2 * dim + 1) * neurons
        # ReLU
        if relu:
            num_flops += 2 * neurons
        self.flops.append((name, num_flops))

        return outputs

    def lrn_layer(self, inputs, name):
        depth_radius=4
        with tf.variable_scope(name) as scope:
            outputs = tf.nn.lrn(inputs,
                                depth_radius=depth_radius,
                                bias=1.0,
                                alpha=0.001/9.0,
                                beta=0.75,
                                name=scope.name)

        input_size = np.prod(inputs.get_shape().as_list()[1:])

        # Evaluate layer operations
        # First, cost to calculate normalizer (using local input squares sum)
        # norm = (1 + alpha/n*sum[n](local-input*local_input)
        local_flops = 1 + 1 + 1 + 2*depth_radius*depth_radius
        # Then cost to divide each input by the normalizer
        num_flops = (local_flops + 1)*input_size
        self.flops.append((name, num_flops))

        return outputs

    def avg_layer(self, inputs, name):
        w = inputs.get_shape().as_list()[1]
        h = inputs.get_shape().as_list()[2]
        c = inputs.get_shape().as_list()[3]
        with tf.variable_scope(name) as scope:
            # Use current spatial dimensions as Kernel size to produce a scalar
            avg = tf.nn.avg_pool(inputs,
                                 ksize=[1,w,h,1],
                                 strides=[1,1,1,1],
                                 padding='VALID',
                                 name=scope.name)
        # Reshape output to remove spatial dimensions reduced to one
        return tf.reshape(avg, shape=[-1,c])

    def inference(self, images):
        raise NotImplementedError('Model subclasses must implement this method')

    def loss(self, logits, labels):

        # Calculate the average cross entropy loss across the batch.
        labels = tf.cast(labels, tf.int64)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits, name='cross_entropy_per_example')
        cross_entropy_loss = tf.reduce_mean(cross_entropy,
                                            name='cross_entropy_loss')
        # We use a global collection to track losses
        tf.add_to_collection('losses', cross_entropy_loss)

        # The total loss is the sum of all losses, including the cross entropy
        # loss and all of the weight losses (see variables declarations)
        total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

        return total_loss

    def accuracy(self, logits, labels):

        # Evaluate predictions
        predictions_op = tf.nn.in_top_k(logits, labels, 1)

        return tf.reduce_mean(tf.cast(predictions_op, tf.float32), name='accuracy')

    def get_flops(self):
        num_flops = 0
        for layer in self.flops:
            num_flops += layer[1]
        return num_flops

    def get_size(self):
        size = 0
        for layer in self.sizes:
            size += layer[1]
        return size