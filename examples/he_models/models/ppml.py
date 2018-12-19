# Python 2 & 3 compatibility
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import model

WEIGHT_DECAY = 1e2

class PPML(model.Model):
    def __init__(self, wd=WEIGHT_DECAY, training=True):

        super(PPML, self).__init__(
            model_name='ppml', wd=wd, training=training)

    def inference(self, images):
        conv1 = self.conv_layer(
            images,
            size=3,
            filters=32,
            stride=1,
            decay=True,
            relu_act=True,
            activation=True,
            bn_before_act=True,
            name='conv1')

        conv2 = self.conv_layer(
            conv1,
            size=3,
            filters=32,
            stride=1,
            decay=True,
            relu_act=True,
            activation=True,
            bn_before_act=True,
            name='conv2')

        pool1 = self.pool_layer(conv2, size=2, stride=2, name='pool1')

        conv3 = self.conv_layer(
            pool1,
            size=3,
            filters=64,
            stride=1,
            decay=True,
            relu_act=True,
            activation=False,
            bn_before_act=True,
            name='conv3')

        conv4 = self.conv_layer(
            conv3,
            size=3,
            filters=64,
            stride=1,
            decay=True,
            relu_act=True,
            activation=True,
            bn_before_act=True,
            name='conv4')

        pool2 = self.pool_layer(conv4, size=2, stride=2, name='pool2')

        fc1 = self.fc_layer(
            pool2,
            neurons=10,
            decay=True,
            relu_act=True,
            activation=True,
            name='fc1')

        return fc1