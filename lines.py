from __future__ import division

import numpy as np
import cv2

from sklearn.cluster import KMeans


def get_line_coords(img):
    lines = cv2.HoughLinesP(img, rho=1, theta=1 * np.pi /
                            180, threshold=100, minLineLength=100, maxLineGap=5)
    return lines


def get_lines_and_params(lines):
    ret = {}
    for i, lin in enumerate(lines):
        for x1, y1, x2, y2 in lin:
            m, b = calculate_line_params([x1, y1, x2, y2])
            ret[i] = [[m, b], [x1, y1, x2, y2]]

    return ret


def clusterize(d):
    data = [[d[key][0][0], d[key][0][1]] for key in d]
    X = np.array(data)

    kmeans = KMeans(n_clusters=2, init='k-means++', max_iter=150).fit(X)
    labels = kmeans.labels_

    for i, key in enumerate(d):
        d[key].append(labels[i])  # d[key][2]

    return d


def calculate_line_params(line):
    x1, y1, x2, y2 = line
    m = (y2 - y1) / (x2 - x1)
    b = -m * x1 + y1
    return m, b


def classify_lines(lines):
    line_class = {0: [], 1: []}

    lines_d = get_lines_and_params(lines)
    clusters_d = clusterize(lines_d)

    for key in clusters_d:
        i = clusters_d[key][2]
        line_class[i].append(clusters_d[key][1])  # line coordinates

    return line_class


def get_final_lines(line_class):

    ret = []
    for i, key in enumerate(line_class):
        if line_class[key]:
            l = get_longest_line(np.array(line_class[i]))
            ret.append(l)
    return ret


def get_longest_line(lines):
    x1 = min(lines[:, 0])
    y1 = max(lines[:, 1])
    x2 = max(lines[:, 2])
    y2 = min(lines[:, 3])
    return [x1, y1, x2, y2]


def longest_line(lines):
    x1 = min(lines[:, 0, 0])
    y1 = max(lines[:, 0, 1])
    x2 = max(lines[:, 0, 2])
    y2 = min(lines[:, 0, 3])
    return [x1, y1, x2, y2]    
