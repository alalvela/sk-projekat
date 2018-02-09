from __future__ import division

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from matplotlib import pyplot as plt
import cv2
import roi_preproccessing as rp
import utils as u


MODEL_PATH = 'cnn_model_bin.h5'


def prepare_bin(image):
    image = cv2.GaussianBlur(image, (5, 5), 0)
    ret = rp.prepr(image)
    return ret


def init_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    kernel = np.ones((3, 3), np.uint8)

#****************
    x_train = [prepare_bin(img) for img in x_train]
    x_test = [prepare_bin(img) for img in x_test]

#****************

    x_train = np.array(x_train)
    x_test = np.array(x_test)

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
          batch_size=32, epochs=12, verbose=1)

score = model.evaluate(test[0], test[1], verbose=0)

print score
model.save(MODEL_PATH)
