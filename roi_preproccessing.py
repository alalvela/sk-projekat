import utils as u
import cv2
import numpy as np
import cnn_predict as p

imga = u.load_image('images/nova.png')


# blr = cv2.GaussianBlur(imga.copy(), (0,0), 3)
# new = cv2.addWeighted(imga.copy(), 1.2, blr, -0.8, 0)

gray = cv2.cvtColor(imga, cv2.COLOR_RGB2GRAY)
# ret, binar = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)
ret, binar = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY)
reta, reg = u.select_min_roi(imga, binar)

predictions = []

# **********************
for i in range(0, len(reg)):
    w, h = reg[i][2][1]
    img = reg[i][0]

    blr = cv2.GaussianBlur(img.copy(), (0, 0), 3)
    img = cv2.addWeighted(img.copy(), 1.1, blr, -0.9, 0)

    if w < 20:
        px = 10 - np.floor(w / 2)
        px = px.astype(np.uint8)
        # dodaj padding na sliku
        img = cv2.copyMakeBorder(
            img, 0, 0, px, px, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    if h < 20:
        py = 10 - np.floor(h / 2)
        py = py.astype(np.uint8)
        # dodaj padding
        img = cv2.copyMakeBorder(
            img, py, py, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    img = cv2.resize(img, (20, 20), interpolation=cv2.INTER_NEAREST)
    img = cv2.copyMakeBorder(
        img, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    img = u.scale_to_range(img)

    pre = p.predict(img)[0].astype(np.uint8)
    predictions.append(pre)

print predictions
# *****************
# img = reg[6][0]

# blr = cv2.GaussianBlur(img.copy(), (0, 0), 3)
# img = cv2.addWeighted(img.copy(), 1.0, blr, -1.0, 0)

# params = reg[6][2]
# x, y, = params[0]
# w, h = params[1]

# if w < 20:
#     px = 10 - np.floor(w / 2)
#     px = px.astype(np.uint8)
#     # dodaj padding na sliku
#     img = cv2.copyMakeBorder(
#         img, 0, 0, px, px, cv2.BORDER_CONSTANT, value=[0, 0, 0])
# if h < 20:
#     py = 10 - np.floor(h / 2)
#     py = py.astype(np.uint8)
#     # dodaj padding
#     img = cv2.copyMakeBorder(
#         img, py, py, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])

# img = cv2.resize(img, (20, 20), interpolation=cv2.INTER_NEAREST)
# img = cv2.copyMakeBorder(img, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=[0, 0, 0])

# u.show_image_gray(img, True)
