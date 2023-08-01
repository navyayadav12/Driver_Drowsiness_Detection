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

yawn_model = keras.models.load_model("./yawn_model.h5")
eye_model = keras.models.load_model("best_model_2.h5")
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


path = os.getcwd()

score = 0

# set a counter
counter = 0


# val = ""

# def load_image(image_file):
# 	img = Image.open(image_file)
# 	return img

def get_label(frame):
	# val = "./" + filename
	# val=str(val)
	# print(val)
	# frame = cv2.imread(val)

	frame = np.array(frame)
	# print('hi')
	image_for_prediction = eye_cropper(frame)
	try:
		image_for_prediction = image_for_prediction / 255.0
	except:
		pass
	# print('got cropped')
	# get prediction from model
	prediction = eye_model.predict(image_for_prediction)
	print(prediction)

	# fac = cv2.cvtColor(fac, cv2.COLOR_BGR2GRAY)
	# fac = cv2.resize(frame, (100, 100))
	# fac = fac / 255
	# fac = fac.reshape(100, 100, -1)
	# fac = np.expand_dims(fac, axis=0)
	# fpred = yawn_model.predict(fac, batch_size=1)
	if prediction >= 0.5:
		val = "sleepy"
	else:
		val = "not sleepy"
	# print(val)
	return val

ans = []
def is_sleepy():
	# print(filename)
	# if filename == 'timg3.jpg':
	img = cv2.imread('./timg3.jpg')
	val = get_label(img)
	ans.append(val)
	# print(val)
	return ans

# pre=is_sleepy()
# print(len(pre))
# print(pre[0])