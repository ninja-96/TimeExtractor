import os
import cv2
import numpy as np

import pickle

from LeNet import LeNet

import torch
import torchvision

from PIL import Image


video_file = 't15'

# cap = cv2.VideoCapture(f'train/{video_file}.mov')
cap = cv2.VideoCapture(f'video/videoplayback.mp4')

d_w = 5
d_h = 6
# [71, 12], [76, 12], [85, 12], [90, 12], [99, 12], [104, 12]
digits = [[8, 12], [12, 12], [21, 12], [26, 12], [35, 12], [40, 12], [44, 12], [49, 12], [71, 12], [76, 12], [85, 12], [90, 12], [99, 12], [104, 12]]

# [x_min, y_min, x_max, y_max]
digit_areas = []
for d in digits:
    digit_areas.append([d[0], d[1], d[0] + d_w, d[1] + d_h])

ret, img = cap.read()
if ret:
    cv2.imwrite('images/img.png', img)

model = LeNet()
model.load_state_dict(torch.load('lenet.pt', map_location='cpu'))
model = model.eval()

k = 0
while True:
    ret, img = cap.read()

    if ret:
        if k % 30 == 0:
            # os.system(f'mkdir ./digits2_raw/{k}')

            for idx, d in enumerate(digit_areas):
                # img = cv2.rectangle(img, (d[0], d[1]), (d[2], d[3]), (255, 255, 255), 1)
                tmp = img[d[1]:d[3], d[0]:d[2]]
                tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)

                tmp = cv2.resize(tmp, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
                print(tmp.shape)
                # tmp = cv2.adaptiveThreshold(tmp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 1)

                cv2.imshow(f'tmp - {idx}', tmp)
                # cv2.imwrite(f'./digits2_raw/{k}/{video_file}_{k}_{idx}.png', tmp)

            # cv2.imwrite(f'./digits2_raw/{video_file}_{k}_main.png', img)

        # k += 1
        cv2.imshow('Frame', img)
        kk = cv2.waitKey(1)

        if kk == 113:
            break
    else:
        print('Error read')
        exit(1)
