import argparse
import json

import cv2
import numpy as np

import pickle

if __name__ == '__main__':
    parser = argparse.ArgumentParser('CNN')
    parser.add_argument('--config', type=str, default='./svm_config.json')
    parser.add_argument('--source', type=str, default='./video/videoplayback.mp4')

    args = parser.parse_args()
    print(args)

    config = json.loads(open(args.config, 'r').read())
    print(config)

    cap = cv2.VideoCapture(args.source)

    # [x_min, y_min, x_max, y_max]
    digit_areas = []
    for d in config['digit_pos']:
        digit_areas.append([d[0], d[1], d[0] + config['d_width'], d[1] + config['d_height']])

    clf = pickle.load(open(config['model'], 'rb'))
    trigger = False

    while True:
        ret, img = cap.read()

        if ret:
            pred = []
            for idx, d in enumerate(digit_areas):
                tmp = img[d[1]:d[3], d[0]:d[2]]
                tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)

                tmp = cv2.resize(tmp, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
                tmp = cv2.adaptiveThreshold(tmp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2)

                trigger = cv2.adaptiveThreshold(tmp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
                if trigger[0, 0] == 0:
                    tmp = cv2.bitwise_not(tmp)

                # cv2.imshow(f'{idx}', tmp)

                tmp = np.true_divide(tmp, 255.0)
                tmp = tmp.flatten().reshape(1, -1)
                pred.append(int(clf.predict(tmp)[0]))

            zoom_img = img[8:20, 0:120, :]

            time = f'{pred[0]}{pred[1]}:{pred[2]}{pred[3]}:{pred[4]}{pred[5]}'
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, time, (10, 200), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow('Frame', img)
            cv2.imshow('Only for humans (zoom)', zoom_img)
            k = cv2.waitKey()

            if k == 113:
                break
        else:
            print('Error read')
            exit(1)
