# Python 2 & 3 compatibility
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import model

WEIGHT_DECAY = 1e-2

class CryptoDL(model.Model):

    def __init__(self, wd=WEIGHT_DECAY, bool_training=False):

        super(CryptoDL, self).__init__(model_name='CryptoDL', wd=wd, bool_training=bool_training)

    def inference(self, images):

        # conv1
        conv1 = self.conv_layer(images,
                                size=5,
                                filters=96,
                                stride=2,
                                decay=True,
                                activation=False,
                                bn=True,
                                name='conv1')

        conv2 = self.conv_layer(conv1,
                                size=5,
                                filters=96,
                                stride=1,
                                decay=True,
                                activation=False,
                                bn=True,
                                name='conv2')

        conv3 = self.conv_layer(conv2,
                        size=5,
                        filters=96,
                        stride=1,
                        decay=True,
                        activation=True,
                        bn=True,
                        name='conv3')

        conv4 = self.conv_layer(conv3,
                        size=5,
                        filters=192,
                        stride=1,
                        decay=True,
                        activation=False,
                        bn=True,
                        name='conv4')

        conv5 = self.conv_layer(conv4,
                        size=5,
                        filters=192,
                        stride=1,
                        decay=True,
                        activation=True,
                        bn=True,
                        name='conv5')

        conv6 = self.conv_layer(conv5,
                        size=5,
                        filters=192,
                        stride=1,
                        decay=True,
                        activation=False,
                        bn=True,
                        name='conv6')

        conv7 = self.conv_layer(conv5,
                        size=5,
                        filters=192,
                        stride=1,
                        decay=True,
                        activation=False,
                        bn=True,
                        name='conv7')

        conv8 = self.conv_layer(conv7,
                        size=5,
                        filters=192,
                        stride=1,
                        decay=True,
                        activation=True,
                        bn=True,
                        name='conv8')


        # local3
        fc2 = self.fc_layer(conv8,
                            neurons=10,
                            decay=True,
                            activation=False,
                            bn=True,
                            name='fc2')

        return fc2