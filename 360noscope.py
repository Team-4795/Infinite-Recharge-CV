from collections import deque
from imutils.video import VideoStream
import numpy as np
import cv2
import imutils
import time
from os import listdir
from os.path import isfile, join
import json
from networktables import NetworkTables
import math

redLower = (0, 20, 210)
redUpper = (10, 50, 255)
blueLower = (70, 20, 210)
blueUpper = (100, 50, 255)
yellowLower = (20, 70, 60)
yellowUpper = (40, 255, 255)
mode = 'ball'

vs = VideoStream(src=2).start()

time.sleep(2.0)

files = [f for f in listdir('./images') if isfile(join('./images', f))]

NetworkTables.initialize(server='roborio-4795-frc.local')
sd = NetworkTables.getTable('SmartDashboard')

while True:
    frame = vs.read()
    '''frame = cv2.imread('./images/' + file)
    data = False
    with open('./labels/' + file + '.json', 'r') as read_file:
        data = json.load(read_file)

    if len(data['objects']) == 0:
        continue'''

    if frame is None:
        break

    frame = imutils.resize(frame, width=600)

    hsv = None
    if mode == 'ball':
        # remove blur from output
        frame = cv2.GaussianBlur(frame, (11, 11), 0)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    if mode == 'red':
        mask = cv2.inRange(hsv, redLower, redUpper)
    elif mode == 'blue':
        mask = cv2.inRange(hsv, blueLower, blueUpper)
    else:
        mask = cv2.inRange(hsv, yellowLower, yellowUpper)
    
    if mode == 'ball':
        mask = cv2.erode(mask, None, iterations=3)
        mask = cv2.dilate(mask, None, iterations=7)
        mask = cv2.erode(mask, None, iterations=2)
    else:
        mask = cv2.dilate(mask, None, iterations=4)
        mask = cv2.GaussianBlur(mask, (41, 41), 0)
        mask = cv2.inRange(mask, 127, 255)
        mask = cv2.dilate(mask, None, iterations=2)
    
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    width = frame.shape[1]
    height = frame.shape[0]
    options = []

    for cnt in cnts:
        if mode == 'ball':
            ((x, y), radius) = cv2.minEnclosingCircle(cnt)
            if cv2.contourArea(cnt) / (radius * radius * math.pi) > 0.5 and radius > 20:
                options.append(cnt)
        else:
            x, y, w, h = cv2.boundingRect(cnt)
            average = mask[y:y+h, x:x+w].mean(axis=0).mean(axis=0)
            if average / 255 < 0.4 and w > 40 and h > 30 and w * 1.1 > h:
                options.append(cnt)

    if len(options) > 0:
        cnt = max(options, key=cv2.contourArea)
        if mode == 'ball':
            ((x, y), radius) = cv2.minEnclosingCircle(cnt)
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 3)
            sd.putNumber('ball_x', x / width * 2 - 1)
            sd.putNumber('ball_y', y / height * 2 - 1)
            sd.putNumber('ball_size', min(radius / width * 5, 1))
            sd.putNumber('target_x', 0)
            sd.putNumber('target_y', 0)
        else:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y, w, h), (0, 255, 255), 3)
            sd.putNumber('target_x', x / width * 2 - 1)
            sd.putNumber('target_y', y / height * 2 - 1)
            sd.putNumber('ball_x', 0)
            sd.putNumber('ball_y', 0)
            sd.putNumber('ball_size', 0)
    else:
        sd.putNumber('ball_x', 0)
        sd.putNumber('ball_y', 0)
        sd.putNumber('ball_size', 0)
        sd.putNumber('target_x', 0)
        sd.putNumber('target_y', 0)
        # calc distance and angle
        # other angle using aspect ratio

    key = cv2.waitKey(1) & 0xFF

    cv2.imshow('Frame', frame)
    cv2.imshow('x', mask)

    #time.sleep(1)

cv2.destroyAllWindows()
