# Python 2 & 3 compatibility
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import model

WEIGHT_DECAY = 5e2

class Tutorial(model.Model):

    def __init__(self, wd=WEIGHT_DECAY, bool_training=False):

        super(Tutorial, self).__init__(model_name='tutorial', wd=wd, bool_training=bool_training)

    def inference(self, images):

        # conv1
        conv1 = self.conv_layer(images,
                                size=5,
                                filters=20,
                                stride=2,
                                decay=False,
                                activation=True,
                                bn=False,
                                name='conv1')

        # pool1 & norm1
        pool1 = self.pool_layer(conv1,
                                size=3,
                                stride=2,
                                name='pool1')

        conv2 = self.conv_layer(conv1,
                                size=5,
                                filters=96,
                                stride=1,
                                decay=False,
                                activation=False,
                                bn=False,
                                name='conv2')
                # pool1 & norm1
        pool2 = self.pool_layer(conv2,
                                size=3,
                                stride=2,
                                name='pool1')


        # local3
        fc2 = self.fc_layer(pool1,
                            neurons=384,
                            decay=True,
                            activation=True,
                            name='fc2')

        # local4
        fc3 = self.fc_layer(fc2,
                            neurons=192,
                            decay=True,
                            name='fc3')


        return fc3