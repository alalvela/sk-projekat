import cv2
import utils as u
import lines as l
import tres as tr
import numpy as np
import vector_ops as v
from scipy import ndimage
import matplotlib.pyplot as plt
import roi_preproccessing as rp
import cnn_predict as p

cc = -1


def nextId():
    global cc
    cc += 1
    return cc


def inRange(r, item, items):
    retVal = []
    for obj in items:
        mdist = v.distance(item['center'], obj['center'])
        if(mdist < r):
            retVal.append(obj)
    return retVal


def get_result_from_video(path):
    cap = cv2.VideoCapture(path)

    elements = []
    t = 0
    counter = 0
    counter_g = 0
    times = []

    passed_blue = {}
    passed_green = {}

    kernel = np.ones((2, 2), np.uint8)
    lower = np.array([230, 230, 230])
    upper = np.array([255, 255, 255])

    while(1):
        ret, img = cap.read()

        if not ret:
            break

        frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        blue_bin = tr.get_blue_line(frame.copy())
        green_bin = tr.get_green_line(frame.copy())

        blue_bin = blue_bin.astype(np.uint8) * 255
        green_bin = green_bin.astype(np.uint8) * 255

        blue_coords = l.get_line_coords(blue_bin)
        b = l.longest_line(blue_coords)
        b = [[b[0], b[1]], [b[2], b[3]]]

        green_coords = l.get_line_coords(green_bin)
        g = l.longest_line(green_coords)
        g = [[g[0], g[1]], [g[2], g[3]]]
        #(lower, upper) = boundaries[0]
        # create NumPy arrays from the boundaries
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")
        mask = cv2.inRange(img, lower, upper)
        img0 = 1.0 * mask

        img0 = cv2.dilate(img0, kernel)  # cv2.erode(img0,kernel)
        # img0 = cv2.dilate(img0, kernel)

        labeled, nr_objects = ndimage.label(img0)
        objects = ndimage.find_objects(labeled)
        # cv2.imshow('frame', labeled)

        for i in range(nr_objects):
            loc = objects[i]

            (xc, yc) = ((loc[1].stop + loc[1].start) / 2,  # slice[1] - x osa
                        (loc[0].stop + loc[0].start) / 2)  # slice[0] - y osa
            (dxc, dyc) = ((loc[1].stop - loc[1].start),
                          (loc[0].stop - loc[0].start))
            (x, y) = (loc[1].start, loc[0].start)

            if(dxc > 11 or dyc > 11):
                reg_img = gray[y: y + dyc + 1, x: x + dxc + 1]
                # cv2.imshow('asd', reg_img)
                cv2.circle(img, (xc, yc), 16, (25, 25, 255), 1)
                elem = {'center': (xc, yc), 'size': (dxc, dyc),
                        't': t}
                # find in range

                lst = inRange(18, elem, elements)
                nn = len(lst)
                if nn == 0:
                    elem['id'] = nextId()
                    elem['t'] = t
                    elem['t0'] = t
                    elem['pass_blue'] = False
                    elem['pass_green'] = False
                    elem['history'] = [
                        {'center': (xc, yc), 'size': (dxc, dyc), 't': t, 'img_raw': reg_img}]
                    elem['future'] = []
                    elements.append(elem)
                elif nn == 1:
                    lst[0]['center'] = elem['center']
                    lst[0]['t'] = t
                    lst[0]['history'].append(
                        {'center': (xc, yc), 'size': (dxc, dyc), 't': t, 'img_raw': reg_img})
                    lst[0]['future'] = []

        for el in elements:
            tt = t - el['t']
            if(tt < 3):
                # ******** BLUE
                dist, pnt, r = v.pnt2line(el['center'], b[0], b[1])
                if r > 0:
                    cv2.line(img, pnt, el['center'], (0, 255, 25), 1)
                    c = (25, 25, 255)
                    if(dist < 10):
                        c = (0, 255, 160)
                        if el['pass_blue'] == False:
                            el['pass_blue'] = True

                            counter += 1

                    cv2.circle(img, el['center'], 16, c, 2)

                # ******** GREEN
                dist, pnt, r = v.pnt2line(el['center'], g[0], g[1])
                if r > 0:
                    cv2.line(img, pnt, el['center'], (0, 255, 25), 1)
                    c = (25, 25, 255)
                    if(dist < 10):
                        c = (0, 255, 160)
                        if el['pass_green'] == False:
                            el['pass_green'] = True

                            counter_g += 1

                    cv2.circle(img, el['center'], 16, c, 2)

                id = el['id']
                cv2.putText(img, str(el['id']),
                            (el['center'][0] + 10, el['center'][1] + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
                for hist in el['history']:
                    ttt = t - hist['t']
                    if(ttt < 100):
                        cv2.circle(img, hist['center'], 1, (0, 255, 255), 1)

                for fu in el['future']:
                    ttt = fu[0] - t
                    if(ttt < 100):
                        cv2.circle(img, (fu[1], fu[2]), 1, (255, 255, 0), 1)

        for el in elements:
            if el['pass_blue'] == True:
                if el['id'] not in passed_blue:
                    passed_blue[el['id']] = el

            if el['pass_green'] == True:
                if el['id'] not in passed_green:
                    passed_green[el['id']] = el

        cv2.putText(img, 'Counter: ' + str(counter), (400, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (90, 90, 255), 2)

        cv2.putText(img, 'Counter: ' + str(counter_g), (400, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (90, 90, 255), 2)

        t += 1

        # cv2.imshow('frame', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    total = get_total(passed_blue, passed_green)
    return total


def get_total(passed_blue, passed_green):

    total = 0

    sum_blue = 0
    sum_green = 0
    proccess_images_blue = []
    proccess_images_green = []
    most_occ = []
    most_occ_green = []

    for key in passed_blue:
        el_history = passed_blue[key]['history']

        images_blue = []
        for i, el in enumerate(el_history):
            if i % 10 == 0:
                images_blue.append(el)

        images_blue = [[el['img_raw'], el['size']] for el in images_blue]
        proccess_images_blue = [rp.preproccess(
            image_blue[0]) for image_blue in images_blue]
        
        predicted_b = [p.predict(img) for img in proccess_images_blue]
        most_common = max(set(predicted_b), key=predicted_b.count)
        most_occ.append(most_common)
        sum_blue += most_common
        print 'SUM BLUE: ' + str(sum_blue)

    for key in passed_green:
        el_history = passed_green[key]['history']

        images_green = []
        for i, el in enumerate(el_history):
            if i % 10 == 0:
                images_green.append(el)

        images_green = [[el['img_raw'], el['size']] for el in images_green]
        proccess_images_green = [rp.preproccess(
            image_green[0]) for image_green in images_green]

        predicted_g = [p.predict(img) for img in proccess_images_green]
        most_common = max(set(predicted_g), key=predicted_g.count)
        most_occ_green.append(most_common)
        sum_green += most_common
        print 'SUM GREEN: ' + str(sum_green)

    total = sum_blue - sum_green
    print total
    return total
