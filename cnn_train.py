from __future__ import division

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from matplotlib import pyplot as plt


MODEL_PATH = 'cnn_model.h5'


def init_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # additional dimension for depth of input /theano
    x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
    x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train /= 255  # normalization
    x_test /= 255

    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)

    return (x_train, y_train), (x_test, y_test)


train, test = init_data()


model = Sequential()

model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(1, 28, 28)))
model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(train[0], train[1],
          batch_size=32, epochs=3, verbose=1)

score = model.evaluate(test[0], test[1], verbose=0)

model.save(MODEL_PATH)
