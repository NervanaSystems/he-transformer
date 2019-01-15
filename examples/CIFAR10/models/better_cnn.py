from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import model

WEIGHT_DECAY = 1e2


class BetterCNN(model.Model):
    def __init__(self,
                 train_poly_act,
                 batch_norm,
                 wd=WEIGHT_DECAY,
                 training=True):
        super(BetterCNN, self).__init__(
            model_name='better_cnn',
            wd=wd,
            training=training,
            train_poly_act=train_poly_act,
            batch_norm=batch_norm)

    def inference(self, images):
        conv1 = self.conv_layer(
            images,
            size=5,
            filters=40,
            stride=1,
            decay=False,
            activation=True,
            batch_norm=self.batch_norm,
            name='conv1')

        pool1 = self.pool_layer(conv1, size=3, stride=2, name='pool1')

        conv2 = self.conv_layer(
            pool1,
            size=3,
            filters=40,
            stride=1,
            decay=False,
            activation=False,
            batch_norm=False,
            name='conv2')

        pool2 = self.pool_layer(conv2, size=3, stride=2, name='pool2')

        fc1 = self.fc_layer(
            pool2,
            neurons=50,
            decay=True,
            activation=True,
            batch_norm=self.batch_norm,
            name='fc1')

        fc2 = self.fc_layer(
            fc1,
            neurons=25,
            decay=True,
            activation=True,
            batch_norm=self.batch_norm,
            name='fc2')

        fc3 = self.fc_layer(
            fc2,
            neurons=10,
            decay=True,
            activation=True,
            batch_norm=self.batch_norm,
            name='fc3')

        return fc3