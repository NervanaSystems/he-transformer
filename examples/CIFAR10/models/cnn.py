# ==============================================================================
#  Copyright 2019 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import model

WEIGHT_DECAY = 1e2


class CNN(model.Model):
    def __init__(self, wd=WEIGHT_DECAY, training=True):

        super(CNN, self).__init__(model_name='cnn', wd=wd, training=training)

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

        fc3 = self.fc_layer(
            conv3, neurons=10, decay=True, activation=False, name='fc3')

        return fc3