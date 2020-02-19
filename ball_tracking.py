from imutils.video import VideoStream
import cv2
import imutils
import time
from os import listdir
from os.path import isfile, join
import json
from networktables import NetworkTables
import math

yellowLower = (25, 160, 70)
yellowUpper = (40, 255, 255)

vs = VideoStream(src=0).start()

time.sleep(2.0)

files = [f for f in listdir("./images") if isfile(join('./images', f))]

NetworkTables.initialize(server='roborio-4795-frc.local')
sd = NetworkTables.getTable('SmartDashboard')

for file in files:
	frame = vs.read()
	frame = cv2.imread('./images/' + file)
	data = False
	with open('./labels/' + file + '.json', "r") as read_file:
		data = json.load(read_file)

	if len(data['objects']) == 0:
		continue

	if frame is None:
		break

	frame = imutils.resize(frame, width=600)
	blurred = cv2.GaussianBlur(frame, (11, 11), 0)
	hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
	#cv2.imshow("Frame", hsv)

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
	options = []
	for cnt in cnts:
		((x, y), radius) = cv2.minEnclosingCircle(cnt)
		if cv2.contourArea(cnt) / (radius * radius * math.pi) > 0.6 and radius > 20:
			options.append(cnt)

	if len(options) > 0:
		cnt = max(options, key=cv2.contourArea)
		((x, y), radius) = cv2.minEnclosingCircle(cnt)
		cv2.circle(frame, (int(x), int(y)), int(radius),
			(0, 255, 255), 2)
		sd.putNumber('ball_x', int(x - width / 2))
		sd.putNumber('ball_y', int(y - height / 2))
		# calc distance and angle

	cv2.imshow("Frame", frame)
	cv2.imshow("x", mask)
	key = cv2.waitKey(1) & 0xFF

	if key == ord("v"):
		time.sleep(15)

	time.sleep(1)

cv2.destroyAllWindows()