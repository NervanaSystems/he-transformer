from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import model

WEIGHT_DECAY = 1e2


class DeeperCNN(model.Model):
    def __init__(self,
                 train_poly_act,
                 batch_norm,
                 wd=WEIGHT_DECAY,
                 training=True):
        super(DeeperCNN, self).__init__(
            model_name='deeper_cnn',
            wd=wd,
            training=training,
            train_poly_act=train_poly_act,
            batch_norm=batch_norm)

    def inference(self, images):
        conv1 = self.conv_layer(
            images,
            size=5,
            filters=40,
            stride=2,
            decay=True,
            activation=True,
            batch_norm=self.batch_norm,
            name='conv1')

        conv2 = self.conv_layer(
            conv1,
            size=5,
            filters=40,
            stride=2,
            decay=True,
            activation=True,
            batch_norm=self.batch_norm,
            name='conv2')

        pool1 = self.pool_layer(conv2, size=5, stride=2, name='pool1')

        conv3 = self.conv_layer(
            pool1,
            size=3,
            filters=80,
            stride=1,
            decay=True,
            activation=True,
            batch_norm=self.batch_norm,
            name='conv3')

        fc1 = self.fc_layer(
            conv3,
            neurons=10,
            decay=True,
            activation=False,
            batch_norm=False,
            name='fc1')

        return fc1