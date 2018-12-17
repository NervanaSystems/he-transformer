# Python 2 & 3 compatibility
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import model

WEIGHT_DECAY = 1e2

class Simple3(model.Model):

    def __init__(self, wd=WEIGHT_DECAY, training=True):

        super(Simple3, self).__init__(model_name='simple3', wd=wd, training=training )

    def inference(self, images):
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
                                activation=True,
                                bn=True,
                                name='conv2')

        fc3 = self.fc_layer(conv2,
                            neurons=10,
                            decay=True,
                            activation=False,
                            bn=False,
                            name='fc3')

        return fc3
