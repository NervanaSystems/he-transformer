# Python 2 & 3 compatibility
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import model

WEIGHT_DECAY = 1e2

class Simple5(model.Model):

    def __init__(self, wd=WEIGHT_DECAY, training=True):

        super(Simple5, self).__init__(model_name='simple5', wd=wd, training=training )

    def inference(self, images):
        conv1 = self.conv_layer(images,
                                size=5,
                                filters=40,
                                stride=2,
                                decay=True,
                                activation=True,
                                bn=True,
                                name='conv1')

        pool1 = self.pool_layer(conv1, size=5, stride=2, name='pool1')

        conv2 = self.conv_layer(pool1,
                                size=3,
                                filters=80,
                                stride=1,
                                decay=True,
                                activation=True,
                                bn=True,
                                name='conv2')

        pool2 = self.pool_layer(conv2, size=3, stride=2, name='pool2')

        conv3 = self.conv_layer(pool2,
                                size=1,
                                filters=120,
                                stride=1,
                                decay=True,
                                activation=False,
                                bn=True,
                                name='conv3')

        #fc1 = self.fc_layer(conv3, neurons=100, decay=True, activation=True, bn=True, name='fc1')

        fc2 = self.fc_layer(conv3,
                            neurons=10,
                            decay=True,
                            activation=False,
                            bn=False,
                            name='fc2')

        return fc2
