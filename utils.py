from __future__ import division
import matplotlib.pyplot as plt

import numpy as np
import cv2


def load_image(path):
    imga = cv2.imread(path)  # ucitavanje slike sa diska
    img = cv2.cvtColor(imga, cv2.COLOR_BGR2RGB)
    return img


def img_to_grayscale(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img_gray


def img_to_bin(img):
    image = img_to_grayscale(img)
    # image = blur_image(img)
    ret, th1 = cv2.threshold(image, 30, 255, cv2.THRESH_BINARY)
    return th1


def blur_image(img):
    blurred = cv2.medianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), 5)
    return blurred


def remove_noise_gray(gray):
    denoised = cv2.fastNlMeansDenoising(
        gray, h=18, searchWindowSize=25, templateWindowSize=11)
    return denoised


def draw_lines(img, lines):
    for i, l in enumerate(lines):
        color = (255, 255, 0) if i % 2 == 0 else (0, 255, 255)
        cv2.line(img, (l[0], l[1]), (l[2], l[3]), color, 2)
    return img


def show_image_gray(img, gray=False):
    if gray == False:
        plt.imshow(img)
    else:
        plt.imshow(img, 'gray')
    plt.show()


def select_roi(image, image_bin):
    img, contours, hierarchy = cv2.findContours(
        image_bin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    regions = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if area > 10 and h > 10 and w < 100 and h < 100:
            region = image_bin[y:y + h + 1, x:x + w + 1]
            # regions.append([resize_region(region), (x, y, w, h)])
            regions.append([region, (x, y, w, h)])
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    regions = sorted(regions, key=lambda item: item[1][0])
    sorted_regions = [region[0] for region in regions]
    return image, sorted_regions


def resize_region(region):
    return cv2.resize(region, (28, 28), interpolation=cv2.INTER_NEAREST)


def scale_to_range(image):
    return image / 255

# def add_padding(image):


def select_min_roi(image, image_bin):
    img, contours, hierarchy = cv2.findContours(
        image_bin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    regions = []
    for contour in contours:
        # rect = cv2.minAreaRect(contour)
        center, dims, angle = cv2.minAreaRect(contour)
        width, height = dims
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if area > 10 and h > 10 and w < 70 and h < 70:
            region = gray[y:y + h + 1, x:x + w + 1]
            # regions.append([resize_region(region), (x, y, w, h)])
            regions.append([region, (x, y, w, h), [center, [w, h]]])
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    regions = sorted(regions, key=lambda item: item[1][0])
    sorted_regions = [region[0] for region in regions]
    return image, regions