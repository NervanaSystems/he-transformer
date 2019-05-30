import tensorflow as tf
from keras.models import Model, model_from_json
from keras.layers import Input, AveragePooling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.merge import concatenate
from keras.layers import Activation, Dense, Reshape


def get_cnn(nb_classes=10):

    input_img = Input(shape=(32, 32, 3), name='input')

    x = Convolution2D(
        10, kernel_size=(2, 2), strides=1, padding='valid',
        activation=None)(input_img)
    x = Activation('relu')(x)
    x = Reshape((31 * 31 * 10, ))(x)
    #x = Flatten()(x)

    x = Dense(100)(x)
    x = Activation('relu')(x)

    x = Dense(10, name='output')(x)
    x = Activation('softmax')(x)

    model = Model(input=input_img, output=[x])
    return model
