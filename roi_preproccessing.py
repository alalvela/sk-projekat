import utils as u
import cv2
import numpy as np
import math
from scipy import ndimage
# import cnn_predict as p


def preproccess(img):

    blr = cv2.GaussianBlur(img, (0, 0), 3)
    img = cv2.addWeighted(img, 1.3, blr, -0.7, 0)
    img = scale(img)
    shx, shy = getBestShift(img)
    shifted = shift(img, shx, shy)
    shifted = u.scale_to_range(shifted)
    shifted = shifted * 1.7
    shifted[shifted > 1] = 1
    # u.show_image_gray(shifted,True)
    return shifted


def shift(img, sx, sy):
    rows, cols = img.shape
    M = np.float32([[1, 0, sx], [0, 1, sy]])
    shifted = cv2.warpAffine(img, M, (cols, rows))
    return shifted


def getBestShift(img):
    cy, cx = ndimage.measurements.center_of_mass(img)

    rows, cols = img.shape
    shiftx = np.round(cols / 2.0 - cx).astype(int)
    shifty = np.round(rows / 2.0 - cy).astype(int)

    return shiftx, shifty


def scale(gray):

    # makni crne ivice
    while np.sum(gray[0]) == 0:
        gray = gray[1:]

    while np.sum(gray[:, 0]) == 0:
        gray = np.delete(gray, 0, 1)

    while np.sum(gray[-1]) == 0:
        gray = gray[:-1]

    while np.sum(gray[:, -1]) == 0:
        gray = np.delete(gray, -1, 1)

    rows, cols = gray.shape

    # 20x20
    if rows > cols:
        factor = 20.0 / rows
        rows = 20
        cols = int(round(cols * factor))
        gray = cv2.resize(gray, (cols, rows))
    else:
        factor = 20.0 / cols
        cols = 20
        rows = int(round(rows * factor))
        gray = cv2.resize(gray, (cols, rows))

    # do 28x28
    colsPadding = (int(math.ceil((28 - cols) / 2.0)),
                   int(math.floor((28 - cols) / 2.0)))
    rowsPadding = (int(math.ceil((28 - rows) / 2.0)),
                   int(math.floor((28 - rows) / 2.0)))
    gray = np.lib.pad(gray, (rowsPadding, colsPadding), 'constant')

    return gray


def preproccess_region(image, size):
    w, h = size

    img = u.remove_noise_gray(image)

    blr = cv2.GaussianBlur(image.copy(), (0, 0), 3)  # ovo je za 33 %
    img = cv2.addWeighted(image.copy(), 1.2, blr, -0.8, 0)

    if w < h:
        px = np.ceil(h / 2) - np.floor(w / 2)
        px = px.astype(np.uint8)
        # dodaj padding na sliku
        img = cv2.copyMakeBorder(
            img, 0, 0, px, px, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    if h < w:
        py = np.ceil(w / 2) - np.floor(h / 2)
        py = py.astype(np.uint8)
        # dodaj padding
        img = cv2.copyMakeBorder(
            img, py, py, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    img = cv2.resize(img, (20, 20), interpolation=cv2.INTER_NEAREST)
    img = cv2.copyMakeBorder(
        img, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    img = u.scale_to_range(img)

    u.show_image_gray(img, True)
    return img
