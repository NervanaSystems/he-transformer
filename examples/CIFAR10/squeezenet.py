#import tensorflow as tf
import numpy as np
#from tensorflow.contrib.layers import Conv2D, avg_pool2d, MaxPooling2D
#from tensorflow.contrib.layers import batch_norm, l2_regularizer
#from tensorflow.contrib.framework import add_arg_scope
#from tensorflow.contrib.framework import arg_scope

import keras

from keras.models import Model, model_from_json
from keras.layers import Input, Concatenate, AveragePooling2D
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.regularizers import l2


def SimpleCNN(num_classes=10):
    # Simple Model ~71% accruacy
    # See https://keras.io/examples/cifar10_cnn/
    input_img = Input(shape=(32, 32, 3), name='input')
    x = Conv2D(32, kernel_size=(3, 3), activation='relu')(input_img)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Reshape((12544, ), input_shape=(-1, 14, 14, 64))(x)
    x = Dense(128, activation='relu')(x)
    # Leave softmax to loss function
    x = Dense(num_classes, name='output')(x)

    model = Model(input=input_img, output=[x])
    return model


def Squeezenet(num_classes=10):
    # Squeeze1 from https://github.com/kaizouman/tensorsandbox/tree/master/cifar10/models/squeeze
    input_img = Input(shape=(32, 32, 3))
    x = Conv2D(
        64, kernel_size=(3, 3), activation='relu', padding='same')(input_img)
    print(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    print(x)
    x = fire_module(x, 32, 64, 64)
    print(x)
    x = fire_module(x, 32, 64, 64)
    print(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    print(x)

    x = fire_module(x, 32, 128, 128)
    print(x)

    # N x 8 x 8 x 256
    x = fire_module(x, 32, 128, 128)
    print(x)

    # Final conv to get ten classes
    # N x 8 x 8 x 10
    x = Conv2D(
        num_classes, kernel_size=(1, 1), padding='same', activation='relu')(x)
    print(x)

    # x = N x 1 x 1 x 10
    # global pooling work-around
    x = AveragePooling2D(pool_size=(8, 8))(x)
    print(x)

    y = Flatten()(x)
    print(y)

    model = Model(input=input_img, output=[y])
    return model


def fire_module(x,
                squeeze_depth,
                expand1_depth,
                expand3_depth,
                reuse=None,
                scope=None):
    x = Conv2D(32, (1, 1), padding="same")(x)
    x = Activation('relu')(x)

    left = Conv2D(expand1_depth, (1, 1), padding='same')(x)
    left = Activation('relu')(left)

    right = ZeroPadding2D(padding=(1, 1))(x)
    right = Conv2D(expand3_depth, (3, 3), padding='valid')(right)
    right = Activation('relu')(right)

    x = keras.layers.concatenate([left, right])
    return x
