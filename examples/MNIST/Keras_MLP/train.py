# ==============================================================================
#  Copyright 2018-2019 Intel Corporation
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
"""An MNIST classifier based on Cryptonets using convolutional layers. """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Activation, Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import losses

import argparse
import sys
import time
import numpy as np
import itertools
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mnist_util import load_mnist_data


def main(FLAGS):
    (x_train, y_train, x_test, y_test) = load_mnist_data()

    print('x_train', x_train.shape)
    print('y_train', y_train.shape)

    model = Sequential()
    model.add(
        Conv2D(
            32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28,
                                                                    1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, name='output'))
    model.add(Activation('softmax'))

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    model.summary()

    model.fit(
        x_train,
        y_train,
        batch_size=FLAGS.batch_size,
        epochs=FLAGS.epochs,
        verbose=1,
        validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    model.save('./model.h5')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--epochs', type=int, default=5, help='Number of training iterations')
    parser.add_argument(
        '--batch_size', type=int, default=128, help='Batch Size')
    FLAGS, unparsed = parser.parse_known_args()

    main(FLAGS)
