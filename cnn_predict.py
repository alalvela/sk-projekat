from keras.models import load_model
import matplotlib.pyplot as plt


MODEL_PATH = 'cnn_model.h5'


def predict(image):
    ret = model.predict_classes(image.reshape((1, 1, 28, 28)))
    return ret



model = load_model(MODEL_PATH)



# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# print '************** LOADED DATA ******************'


# plt.imshow(x_test[10])
# plt.show()


# sc = model.predict_classes(x_test[10].reshape((1, 1, 28, 28)))
# print 'CLASS:' + str(sc)
