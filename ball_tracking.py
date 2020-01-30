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

yellowLower = (15, 160, 60)
yellowUpper = (30, 255, 255)

vs = VideoStream(src=0).start()

time.sleep(2.0)

files = [f for f in listdir("./images") if isfile(join('./images', f))]

NetworkTables.initialize(server='roborio-4795-frc.local')
sd = NetworkTables.getTable('SmartDashboard')

while True:#for file in files:
	frame = vs.read()
	'''frame = cv2.imread('./images/' + file)#frame[1] if args.get("video", False) else frame
	data = False
	with open('./labels/' + file + '.json', "r") as read_file:
		data = json.load(read_file)

	if len(data['objects']) == 0:
		continue'''

	if frame is None:
		break

	frame = imutils.resize(frame, width=600)
	blurred = cv2.GaussianBlur(frame, (11, 11), 0)
	hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

	mask = cv2.inRange(hsv, yellowLower, yellowUpper)
	mask = cv2.erode(mask, None, iterations=3)
	mask = cv2.dilate(mask, None, iterations=7)
	mask = cv2.erode(mask, None, iterations=2)

	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	center = None
	width = frame.shape[1]
	height = frame.shape[0]

	if len(cnts) > 0:
		c = max(cnts, key=cv2.contourArea)
		((x, y), radius) = cv2.minEnclosingCircle(c)
		M = cv2.moments(c)

		if radius > 10:
			cv2.circle(frame, (int(x), int(y)), int(radius),
				(0, 255, 255), 2)
			sd.putNumber('ball_x', int(x - width / 2))
			sd.putNumber('ball_y', int(y - height / 2))

	cv2.imshow("Frame", frame)
	#cv2.imshow("x", mask)
	key = cv2.waitKey(1) & 0xFF

	if key == ord("v"):
		time.sleep(15)

	#time.sleep(1)

# if we are not using a video file, stop the camera video stream
if not args.get("video", False):
	vs.stop()

# otherwise, release the camera
else:
	vs.release()

cv2.destroyAllWindows()