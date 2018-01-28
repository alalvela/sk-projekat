import utils as u
import cv2
import numpy as np
import scipy as sc


def erode_dilate(res):
    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(res, kernel, iterations=1)
    kernel1 = np.ones((5, 5), np.uint8)
    dilation = cv2.dilate(erosion, kernel1, iterations=1)
    return dilation


def dil_er_dil(res):
    kernel = np.ones((3, 3), np.uint8)
    dilation = cv2.dilate(res, kernel, iterations=1)
    erosion = cv2.erode(dilation, kernel, iterations=1)
    kernel1 = np.ones((5, 5), np.uint8)
    dilation = cv2.dilate(erosion, kernel1, iterations=1)
    return dilation


def get_blue_line(img):
    blue = img[:, :, 2] > 200
    mask = img[:, :, 0] < 100
    res = sc.logical_and(blue, mask)
    res = res.astype(np.float32)
    # ret = erode_dilate(res)
    ret = dil_er_dil(res)
    return ret

def get_green_line(img):
    green = img[:, :, 1] > 200
    mask = img[:, :, 0] < 50
    res = sc.logical_and(green, mask)
    res = res.astype(np.float32)
    # ret = erode_dilate(res)
    ret = dil_er_dil(res)
    return ret

# img = u.load_image('images/blueline.png')


# ret = get_blue_line(img)
# u.show_image_gray(ret, True)

