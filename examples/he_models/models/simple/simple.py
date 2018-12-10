# Python 2 & 3 compatibility
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import model

WEIGHT_DECAY = 1e2

class Simple(model.Model):

    def __init__(self, wd=WEIGHT_DECAY, training=False):

        super(Simple, self).__init__(model_name='simple', wd=wd, training=training)

    def inference(self, images):
        #images = tf.identity(images, name='input')

        # conv1
        conv1 = self.conv_layer(images,
                                size=5,
                                filters=40,
                                stride=2,
                                decay=True,
                                activation=False,
                                bn=True,
                                name='conv1')

        fc3 = self.fc_layer(conv1,
                            neurons=10,
                            decay=True,
                            activation=False,
                            bn=False,
                            name='fc3')

        #output = tf.identity(fc3, name='output')

        return fc3