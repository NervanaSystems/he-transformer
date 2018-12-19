# Python 2 & 3 compatibility
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import model

WEIGHT_DECAY = 1e1

class Simple9(model.Model):
    def __init__(self, wd=WEIGHT_DECAY, training=True):

        super(Simple9, self).__init__(
            model_name='simple9', wd=wd, training=training, train_poly_act=False)

    def inference(self, images):
        conv1 = self.conv_layer(
            images,
            size=5,
            filters=40,
            stride=2,
            decay=True,
            activation=True,
            bn_before_act=True,
            name='conv1')

        conv2 = self.conv_layer(
            conv1,
            size=5,
            filters=40,
            stride=2,
            decay=True,
            activation=True,
            bn_before_act=True,
            name='conv2')

        pool1 = self.pool_layer(conv2, size=5, stride=2, name='pool1')

        conv3 = self.conv_layer(
            pool1,
            size=3,
            filters=80,
            stride=1,
            decay=True,
            activation=True,
            bn_before_act=True,
            name='conv3')

        conv4 = self.conv_layer(
            conv3,
            size=3,
            filters=80,
            stride=1,
            decay=True,
            activation=True,
            bn_before_act=True,
            name='conv4')

        fc1 = self.fc_layer(
            conv4,
            neurons=10,
            decay=True,
            activation=False,
            name='fc1')

        return fc1