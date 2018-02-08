from keras.models import load_model
from keras.datasets import mnist
import matplotlib.pyplot as plt
import cv2
import utils as u
import novi as t

# ?*************************

MODEL_PATH = 'cnn_model.h5'


def predict(image):
    ret = model.predict_classes(image.reshape((1, 1, 28, 28)))
    return ret[0]



model = load_model(MODEL_PATH)
# ?*************************





# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# ret = t.get_images()

# train_show = x_train[:10]

# u.show_images(train_show)
# u.show_images(ret[:10])

# # test_show = x_test[:10]
# u.show_images(test_show)

