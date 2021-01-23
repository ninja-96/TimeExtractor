import cv2
import numpy as np

import pickle

cap = cv2.VideoCapture(f'video/videoplayback.mp4')

d_w = 5
d_h = 6
digits = [[71, 12], [76, 12], [85, 12], [90, 12], [99, 12], [104, 12]]

# [x_min, y_min, x_max, y_max]
digit_areas = []
for d in digits:
    digit_areas.append([d[0], d[1], d[0] + d_w, d[1] + d_h])

clf = pickle.load(open('SVC.ski', 'rb'))

while True:
    ret, img = cap.read()

    if ret:
        pred = []
        for idx, d in enumerate(digit_areas):
            tmp = img[d[1]:d[3], d[0]:d[2]]
            tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)

            tmp = cv2.resize(tmp, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
            tmp = cv2.adaptiveThreshold(tmp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

            tmp = np.true_divide(tmp, 255.0)
            tmp = tmp.flatten().reshape(1, -1)
            pred.append(int(clf.predict(tmp)[0]))

        zoom_img = img[8:20, 0:120, :]

        time = f'{pred[0]}{pred[1]}:{pred[2]}{pred[3]}:{pred[4]}{pred[5]}'
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, time, (10, 200), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Frame', img)
        cv2.imshow('Zoom', zoom_img)
        k = cv2.waitKey(1)

        if k == 113:
            break
    else:
        print('Error read')
        exit(1)
