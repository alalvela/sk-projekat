from keras.models import load_model
from keras.datasets import mnist
import matplotlib.pyplot as plt
import cv2
import utils as u
import numpy as np
import roi_preproccessing as rp


# def prepare_bin(image):
#     ret, image = cv2.threshold(image, 120, 255, cv2.THRESH_BINARY)
#     img = t.scale(image)
#     shx, shy = t.getBestShift(img)
#     shifted = t.shift(img, shx, shy)
#     shifted = u.scale_to_range(shifted)
#     return shifted



MODEL_PATH = 'cnn_model_2.h5'


def predict(image):
    ret = model.predict_classes(image.reshape((1, 1, 28, 28)))
    return ret[0]


model = load_model(MODEL_PATH)

