# Python 2 & 3 compatibility
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import model

WEIGHT_DECAY = 1e-2

class Simple(model.Model):

    def __init__(self, wd=WEIGHT_DECAY, bool_training=False):

        super(Simple, self).__init__(model_name='simple', wd=wd, bool_training=bool_training)

    def inference(self, images):

        # conv1
        conv1 = self.conv_layer(images,
                                size=5,
                                filters=20,
                                stride=2,
                                decay=True,
                                activation=False,
                                bn=False,
                                name='conv1')

        conv2 = self.conv_layer(conv1,
                                size=5,
                                filters=50,
                                stride=1,
                                decay=True,
                                activation=True,
                                bn=False,
                                name='conv2')
        fc2 = self.fc_layer(conv2,
                            neurons=500,
                            decay=True,
                            activation=True,
                            bn=False,
                            name='fc2')

        fc3 = self.fc_layer(fc2,
                            neurons=10,
                            decay=True,
                            activation=False,
                            bn=False,
                            name='fc3')

        return fc3