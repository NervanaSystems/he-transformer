from keras.callbacks import ModelCheckpoint
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
import numpy as np
import models
import os

# training setup from keras/examples.

batch_size = 32
nb_classes = 10
nb_epoch = 200

img_rows, img_cols = 32, 32
img_channels = 3

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_train.shape[0], 'test samples')

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

print(x_train[0])

model = models.get_cnn(10)

print(model.summary())

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

model.fit(x_train, y_train, epochs=1)
test_loss, test_acc = model.evaluate(x_test, y_test)

print('Test accuracy:', test_acc)

os.makedirs('./model', exist_ok=True)
model.save('./model/keras_model.h5')
