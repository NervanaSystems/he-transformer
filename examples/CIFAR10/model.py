# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# =============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os

WEIGHT_DECAY = 1e2


class Model(object):
    def __init__(self,
                 model_name,
                 wd=WEIGHT_DECAY,
                 training=True,
                 batch_norm=True,
                 train_poly_act=True):

        self.wd = wd
        self.sizes = []
        self.multiplcative_depth = 0
        self.training = training
        self.model_name = model_name
        self.train_poly_act = train_poly_act
        self.batch_norm = batch_norm
        print("Creating model with decay", wd)

    def _name_to_filename(self, var_name):
        """Given a variable name, returns the filename where to store it"""
        var_name = var_name.replace('/', '_')
        var_name = var_name.replace(':0', '') + '.txt'
        prefix = 'weights'
        model_name = self.model_name
        if self.batch_norm:
            model_name += '_bn'
        if self.train_poly_act:
            model_name += '_train_poly_act'

        return '/'.join([prefix, model_name, var_name])

    def _poly_act(self, x, scope):
        self.multiplcative_depth += 2
        print('poly activation, => mult. depth', self.multiplcative_depth)

        if self.train_poly_act:
            a = self._get_weights_var(
                'a', [], initializer=tf.initializers.zeros, scope=scope)
            b = self._get_weights_var(
                'b', [], initializer=tf.initializers.ones, scope=scope)

            return a * x * x + b * x

        else:
            return 0.125 * x * x + 0.5 * x + 0.25

    def _get_weights_var(self,
                         name,
                         shape,
                         decay=False,
                         scope='',
                         initializer=tf.contrib.layers.xavier_initializer(
                             uniform=False, dtype=tf.float32)):
        if self.training:
            var = tf.get_variable(
                name=name,
                shape=shape,
                initializer=initializer,
                dtype=tf.float32)
            if decay:
                weight_decay = self.wd
                for x in shape:
                    weight_decay /= int(x)
                # Weight loss is L2 loss multiplied by weight decay
                weight_loss = tf.multiply(
                    tf.nn.l2_loss(var), weight_decay, name='weight_loss')
                tf.add_to_collection('losses', weight_loss)

            return var
        else:
            restore_filename = self._name_to_filename(scope.name + '/' + name)
            print('restoring variable: ', restore_filename, ' with shape',
                  shape)

            if os.path.exists('./' + restore_filename):
                return tf.constant(
                    np.loadtxt(restore_filename,
                               dtype=np.float32).reshape(shape))
            else:
                print('Could not load ', restore_filename)
                exit(1)

    def conv_layer(self,
                   inputs,
                   size,
                   filters,
                   stride,
                   decay,
                   name,
                   activation=True,
                   relu_act=False,
                   batch_norm=False):
        channels = inputs.get_shape()[3]
        shape = [size, size, channels, filters]

        if (relu_act and not activation):
            print("Error: relu_act=True, activation=False")
            exit(1)

        self.multiplcative_depth += 1
        print('conv layer => mult. depth', self.multiplcative_depth)

        print('Conv shape: size:', size, 'channels:', channels, 'filters:',
              filters, ', shape: ', shape)

        with tf.variable_scope(name) as scope:
            weights = self._get_weights_var(
                'weights', shape=shape, decay=decay, scope=scope)

            biases = self._get_weights_var(
                'biases',
                shape=[filters],
                initializer=tf.constant_initializer(0.0),
                scope=scope)
            conv = tf.nn.conv2d(
                inputs,
                weights,
                strides=[1, stride, stride, 1],
                padding='SAME')

            if batch_norm:
                conv = tf.layers.batch_normalization(
                    conv, training=self.training)
            outputs = tf.nn.bias_add(conv, biases)

            if activation:
                if relu_act:
                    outputs = tf.nn.relu(outputs)
                else:
                    outputs = self._poly_act(outputs, scope=scope)

        # Evaluate layer size
        self.sizes.append((name, (1 + size * size * int(channels)) * filters))

        return outputs

    def pool_layer(self, inputs, size, stride, name, max_pool=False):
        self.multiplcative_depth += 1
        print('pool layer => mult. depth', self.multiplcative_depth)
        if max_pool:
            print("Max pooling")
            with tf.variable_scope(name) as scope:
                outputs = tf.nn.max_pool(
                    inputs,
                    ksize=[1, size, size, 1],
                    strides=[1, stride, stride, 1],
                    padding='SAME',
                    name=name)
        else:
            with tf.variable_scope(name) as scope:
                outputs = tf.nn.avg_pool(
                    inputs,
                    ksize=[1, size, size, 1],
                    strides=[1, stride, stride, 1],
                    padding='SAME',
                    name=name)

        return outputs

    def fc_layer(self,
                 inputs,
                 neurons,
                 decay,
                 name,
                 activation=True,
                 batch_norm=False,
                 relu_act=False):
        self.multiplcative_depth += 1
        print('FC layer => mult. depth', self.multiplcative_depth)

        with tf.variable_scope(name) as scope:
            if len(inputs.get_shape().as_list()) > 2:
                # We need to reshape inputs:
                # [ batch size , w, h, c ] -> [ batch size, w x h x c ]
                # Batch size is a dynamic value, but w, h and c are static and
                # can be used to specifiy the reshape operation
                dim = np.prod(inputs.get_shape().as_list()[1:])
                reshaped = tf.reshape(inputs, shape=[-1, dim], name='reshaped')
            else:
                # No need to reshape inputs
                reshaped = inputs
            dim = reshaped.get_shape().as_list()[1]
            weights = self._get_weights_var(
                'weights', shape=[dim, neurons], decay=decay, scope=scope)
            biases = self._get_weights_var(
                'biases',
                shape=[neurons],
                initializer=tf.constant_initializer(0.0),
                scope=scope)
            x = tf.add(tf.matmul(reshaped, weights), biases)

            if batch_norm:
                x = tf.layers.batch_normalization(x, training=self.training)
            if activation:
                if relu_act:
                    x = tf.nn.relu(x)
                else:
                    x = self._poly_act(x, scope=scope)

        # Evaluate layer size
        self.sizes.append((name, (dim + 1) * neurons))

        return x

    def avg_layer(self, inputs, name):
        print('avg layer => mult. depth', self.multiplcative_depth)
        self.multiplcative_depth += 1

        w = inputs.get_shape().as_list()[1]
        h = inputs.get_shape().as_list()[2]
        c = inputs.get_shape().as_list()[3]
        with tf.variable_scope(name) as scope:
            # Use current spatial dimensions as Kernel size to produce a scalar
            avg = tf.nn.avg_pool(
                inputs,
                ksize=[1, w, h, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name=scope.name)
        # Reshape output to remove spatial dimensions reduced to one
        return tf.reshape(avg, shape=[-1, c])

    def inference(self, images):
        raise NotImplementedError(
            'Model subclasses must implement this method')

    def loss(self, logits, labels):
        # Calculate the average cross entropy loss across the batch.
        labels = tf.cast(labels, tf.int64)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits, name='cross_entropy_per_example')
        cross_entropy_loss = tf.reduce_mean(
            cross_entropy, name='cross_entropy_loss')
        tf.add_to_collection('losses', cross_entropy_loss)

        total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

        return total_loss

    def accuracy(self, logits, labels):
        predictions_op = tf.nn.in_top_k(logits, labels, 1)

        return tf.reduce_mean(
            tf.cast(predictions_op, tf.float32), name='accuracy')

    def mult_depth(self):
        return self.multiplcative_depth
