import cv2
import utils as ut
import lines as l

cap = cv2.VideoCapture('videos/video-3.avi')

while cap.isOpened():
    ret, frame = cap.read()

    image_bin = ut.img_to_bin(frame)

    lines = l.get_line_coords(image_bin)
    classes = l.classify_lines(lines)
    l1, l2 = l.get_final_lines(classes)
    ret = ut.draw_lines(frame, [l1, l2])

    ret = ut.select_roi(ret, image_bin)
    
    cv2.imshow('frame', ret)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
