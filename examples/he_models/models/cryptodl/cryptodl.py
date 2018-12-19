# Python 2 & 3 compatibility
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import model

WEIGHT_DECAY = 1e-2


class CryptoDL(model.Model):
    def __init__(self, wd=WEIGHT_DECAY, training=False):

        super(CryptoDL, self).__init__(
            model_name='cryptodl',
            wd=wd,
            training=training,
            train_poly_act=True)

    def inference(self, images):

        conv1 = self.conv_layer(
            images,
            size=3,
            filters=96,
            stride=1,
            decay=True,
            activation=False,
            bn_before_act=True,
            name='conv1')

        conv2 = self.conv_layer(
            conv1,
            size=3,
            filters=96,
            stride=1,
            decay=True,
            activation=True,
            bn_before_act=True,
            name='conv2')

        pool1 = self.pool_layer(conv2, size=3, stride=1, name='pool1')

        conv3 = self.conv_layer(
            pool1,
            size=3,
            filters=192,
            stride=1,
            decay=True,
            activation=False,
            bn_before_act=True,
            name='conv4')

        conv4 = self.conv_layer(
            conv3,
            size=3,
            filters=192,
            stride=1,
            decay=True,
            activation=True,
            bn_before_act=True,
            name='conv5')

        pool2 = self.pool_layer(conv4, size=3, stride=1, name='pool1')

        conv5 = self.conv_layer(
            pool2,
            size=3,
            filters=192,
            stride=1,
            decay=True,
            activation=False,
            bn_before_act=True,
            name='conv6')

        conv6 = self.conv_layer(
            conv5,
            size=3,
            filters=192,
            stride=1,
            decay=True,
            activation=True,
            bn_before_act=True,
            name='conv8')

        pool2 = self.pool_layer(conv6, size=3, stride=1, name='pool2')

        fc1 = self.fc_layer(
            pool2,
            neurons=256,
            decay=True,
            activation=False,
            name='fc1')

        fc2 = self.fc_layer(
            fc1,
            neurons=10,
            decay=True,
            activation=False,
            name='fc2')

        return fc2
