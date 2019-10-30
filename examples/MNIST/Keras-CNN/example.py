'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

# https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py

from __future__ import print_function
import sys
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from tensorflow.python.keras.backend import set_session

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mnist_util import load_mnist_data, \
                       get_variable, \
                       conv2d_stride_2_valid, \
                       str2bool, \
                       server_argument_parser, \
                       server_config_from_flags


def load_mnist_data():
    num_classes = 10

    # input image dimensions
    img_rows, img_cols = 28, 28

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return x_train, y_train, x_test, y_test, input_shape


def train(FLAGS):
    x_train, y_train, x_test, y_test, input_shape = load_mnist_data()

    num_classes = 10

    model = Sequential()
    model.add(
        Conv2D(
            32,
            kernel_size=(3, 3),
            activation='relu',
            input_shape=input_shape,
            name='input_1'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adadelta(),
        metrics=['accuracy'])

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

    model.save('keras_cnn.h5')


def test(FLAGS):
    import ngraph_bridge
    config = server_config_from_flags(FLAGS, 'input_1_input')
    sess = tf.compat.v1.Session(config=config)
    set_session(sess)

    x_train, y_train, x_test, y_test, input_shape = load_mnist_data()

    x_test = x_test[0:FLAGS.batch_size]
    y_test = y_test[0:FLAGS.batch_size]

    print(x_test.shape)
    print(y_test.shape)

    model = load_model('keras_cnn.h5', compile=True)
    model.summary()

    y_out = model.predict(x_test, verbose=1)
    print('y_out', y_out)

    y_pred = np.argmax(y_out, 1)
    y_truth = np.argmax(y_test, 1)
    print('y_pred', y_pred)
    print('y_truth', y_truth)
    correct_prediction = np.equal(y_pred, y_truth)
    error_count = np.size(correct_prediction) - np.sum(correct_prediction)
    test_accuracy = np.mean(correct_prediction)

    print('Error count', error_count, 'of', FLAGS.batch_size, 'elements.')
    print('Accuracy: %g ' % test_accuracy)


if __name__ == '__main__':
    parser = server_argument_parser()
    parser.add_argument(
        '--epochs', type=int, default=12, help='Number of training iterations')
    parser.add_argument(
        '--test',
        type=bool,
        default=False,
        help='Whether train (False) or test (True)')

    FLAGS, unparsed = parser.parse_known_args()

    if unparsed:
        print('Unparsed flags:', unparsed)
    if FLAGS.encrypt_server_data and FLAGS.enable_client:
        raise Exception(
            "encrypt_server_data flag only valid when client is not enabled. Note: the client can specify whether or not to encrypt the data using 'encrypt' or 'plain' in the configuration map"
        )

    if not FLAGS.test:
        train(FLAGS)
    else:
        test(FLAGS)