import cv2
import utils as ut
import lines as l
import tres as t
import numpy as np

cap = cv2.VideoCapture('videos/video-9.avi')

cnt = 0

while cap.isOpened():
    ret, frame = cap.read()
    cnt += 1
    
    if not ret:
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    blue_bin = t.get_blue_line(frame)
    green_bin = t.get_green_line(frame)

    blue_bin = blue_bin.astype(np.uint8) * 255
    green_bin = green_bin.astype(np.uint8) * 255

    blue_coords = l.get_line_coords(blue_bin)
    b = l.longest_line(blue_coords)
    print b

    green_coords = l.get_line_coords(green_bin)
    g = l.longest_line(green_coords)
    print g


    lin = ut.draw_lines(frame, [b, g])
    
    # print blue_coords

    # image_bin = ut.img_to_bin(frame)

    # lines = l.get_line_coords(image_bin)
    # classes = l.classify_lines(lines)
    # l1, l2 = l.get_final_lines(classes)
    # ret = ut.draw_lines(frame, [l1, l2])
    lin2bin = ut.img_to_bin(lin)
    ret = ut.select_roi(lin, lin2bin)

    if cnt == 40:
        break
    cv2.imshow('frame', lin)
    # cv2.imshow('frame', blue_bin)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
