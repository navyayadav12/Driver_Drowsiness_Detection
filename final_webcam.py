import requests
import os
import time
import numpy as np
import cv2
# from playsound import playsound
from PIL import Image, ImageDraw
import face_recognition
from tensorflow import keras
from pygame import mixer

mixer.init()
mixer.music.load("emergency-alarm-with-reverb-29431.mp3")
mixer.music.set_volume(0.7)

import pyttsx3

engine = pyttsx3.init()


eye_model = keras.models.load_model("best_model_2.h5")
yawn_model = keras.models.load_model("yawn_model.h5")
# yawn_model = keras.models.load_model("./yawn_new.h5",compile=False)
# yawn_model = keras.models.load_model("./aug_yawn_model.h5",compile=False)

face = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('./haarcascades/haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('./haarcascades/haarcascade_righteye_2splits.xml')
eyes = cv2.CascadeClassifier('haarcascades\haarcascade_eye.xml')


def eye_cropper(frame):
	# create a variable for the facial feature coordinates
	facial_features_list = face_recognition.face_landmarks(frame)

	# create a placeholder list for the eye coordinates
	# and append coordinates for eyes to list unless eyes
	# weren't found by facial recognition
	try:
		eye = facial_features_list[0]['left_eye']
	except:
		try:
			eye = facial_features_list[0]['right_eye']
		except:
			return

	# establish the max x and y coordinates of the eye
	x_max = max([coordinate[0] for coordinate in eye])
	x_min = min([coordinate[0] for coordinate in eye])
	y_max = max([coordinate[1] for coordinate in eye])
	y_min = min([coordinate[1] for coordinate in eye])

	# establish the range of x and y coordinates
	x_range = x_max - x_min
	y_range = y_max - y_min

	# in order to make sure the full eye is captured,
	# calculate the coordinates of a square that has a
	# 50% cushion added to the axis with a larger range and
	# then match the smaller range to the cushioned larger range
	if x_range > y_range:
		right = round(.5 * x_range) + x_max
		left = x_min - round(.5 * x_range)
		bottom = round((((right - left) - y_range)) / 2) + y_max
		top = y_min - round((((right - left) - y_range)) / 2)
	else:
		bottom = round(.5 * y_range) + y_max
		top = y_min - round(.5 * y_range)
		right = round((((bottom - top) - x_range)) / 2) + x_max
		left = x_min - round((((bottom - top) - x_range)) / 2)

	# crop the image according to the coordinates determined above
	cropped = frame[top:(bottom + 1), left:(right + 1)]

	# resize the image
	cropped = cv2.resize(cropped, (80, 80))
	image_for_prediction = cropped.reshape(-1, 80, 80, 3)

	return image_for_prediction


lbl = ['Close', 'Open']
path = os.getcwd()
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count = 0
score = 0
thicc = 2
# rpred = [99]
# lpred = [99]
fpred = [99]
cap = cv2.VideoCapture(0)
w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
if not cap.isOpened():
	raise IOError('Cannot open webcam')

# set a counter
counter = 0
while True:
	ret, frame = cap.read()
	height, width = frame.shape[:2]
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	faces = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
	left_eye = leye.detectMultiScale(gray)
	right_eye = reye.detectMultiScale(gray)
	eye = eyes.detectMultiScale(gray)  # eye detection

	frame_count = 0
	if frame_count == 0:
		frame_count += 1
		pass
	else:
		count = 0
		continue

	# function called on the frame
	image_for_prediction = eye_cropper(frame)
	try:
		image_for_prediction = image_for_prediction / 255.0
	except:
		continue

	# get prediction from model
	prediction = eye_model.predict(image_for_prediction)

	# cv2.rectangle(frame, (0, height - 50), (200, height), (0, 0, 0), 15)
	for (x, y, w, h) in faces:
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
	for (x, y, w, h) in eye:
		cv2.rectangle(frame, (x, y), (x + w + 5, y + h + 5), (0, 255, 0), 2)
	for (x, y, w, h) in faces:
		# cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)
		fac = frame[y:y + h, x:x + w]
		count = count + 1
		fac = cv2.cvtColor(fac, cv2.COLOR_BGR2GRAY)
		fac = cv2.resize(fac, (100, 100))
		fac = fac / 255
		fac = fac.reshape(100, 100, -1)
		fac = np.expand_dims(fac, axis=0)
		fpred = yawn_model.predict(fac, batch_size=1)
	# print(fpred)

	if prediction < 0.5:
		score = score - 1
	else:
		score = score + 1
	# else:
	# 	score = score + 1
	# if prediction < 0.5:
	# 	score = score - 1
	# else:
	# 	score = score + 1
	# print(fpred)
	# if fpred < 0.5:
	# 	score = score - 1
	# else:
	# 	score = score + 1

	if score < 0:
		score = 0
	# print(score)

	cv2.putText(frame, 'Score:' + str(score), (250, 30), font, 2, (255, 55, 50), 2, cv2.LINE_AA)
	if (score >= 0 and score <= 5):
		cv2.putText(frame, "State: Alert", (10, height - 20), font, 1, (0, 255, 0), 1, cv2.LINE_AA)
		cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)
		cv2.imshow('frame', frame)
		mixer.music.stop()
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	elif (score > 5 and score <= 10):
		cv2.putText(frame, "State: Neither Alert Nor Drowsy", (10, height - 20), font, 1, (0, 255, 155), 1,
		            cv2.LINE_AA)
		cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)
		cv2.imshow('frame', frame)
		mixer.music.stop()
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	elif (score > 10 and score <= 20):
		cv2.putText(frame, "State: Sleepy", (10, height - 20), font, 1, (0, 255, 255), 1, cv2.LINE_AA)
		cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)
		cv2.imshow('frame', frame)
		mixer.music.stop()
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	else:
		if (score > 20):
			cv2.putText(frame, "State: Extremely Sleepy", (10, height - 20), font, 1, (255, 0, 0), 1, cv2.LINE_AA)
			# person is feeling sleepy so we beep the alarm
			cv2.imwrite(os.path.join(path, 'image.jpg'), frame)
			try:

				text = "Wake up Sir!"
				engine.say(text)
				engine.runAndWait()
				mixer.music.play()

			except:
				isplaying = False
				pass
			if (thicc < 16):
				thicc = thicc + 2
			else:
				thicc = thicc - 2
				if (thicc < 2):
					thicc = 2
			cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)
		else:
			mixer.music.stop()
		cv2.imshow('frame', frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

cap.release()
cv2.destroyAllWindows()
