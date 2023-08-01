import requests
import os
import time
import cv2
import numpy as np

from PIL import Image, ImageDraw
import face_recognition
from tensorflow import keras
from pygame import mixer

from pygame import mixer

mixer.init()

mixer.music.load("emergency-alarm-with-reverb-29431.mp3")

mixer.music.set_volume(0.7)
eye_model = keras.models.load_model('eye_model.h5')
# eye_model = keras.models.load_model('aug_eye_model.h5', compile=False)
yawn_model = keras.models.load_model("./yawn_model.h5")


# eye_model = keras.models.load_model("./eye_model.h5")


def download_file_from_google_drive(id, destination):
	URL = "https://docs.google.com/uc?export=download"

	session = requests.Session()

	response = session.get(URL, params={'id': id}, stream=True)
	token = get_confirm_token(response)

	if token:
		params = {'id': id, 'confirm': token}
		response = session.get(URL, params=params, stream=True)

	save_response_content(response, destination)


def get_confirm_token(response):
	for key, value in response.cookies.items():
		if key.startswith('download_warning'):
			return value

	return None


def save_response_content(response, destination):
	CHUNK_SIZE = 32768

	with open(destination, "wb") as f:
		for chunk in response.iter_content(CHUNK_SIZE):
			if chunk:  # filter out keep-alive new chunks
				f.write(chunk)


def show(img):
	cv2.imshow("img", img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


face_cascade = cv2.CascadeClassifier(
	'./haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_eye.xml')


def get_cropped_image(img):
	try:
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		faces = face_cascade.detectMultiScale(gray, 1.3, 5)
		(x, y, w, h) = faces[0]
		crop_face = gray[y:y + h, x:x + w]
		crop_face = cv2.resize(crop_face, (100, 100))
		crop_face = np.array(crop_face)
		crop_face = crop_face.astype("float32")
		crop_face /= 255
		eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
		(x1, y1, w1, h1) = eyes[0]
		(x2, y2, w2, h2) = eyes[1]
		crop_eyes = [gray[y1:y1 + h1, x1:x1 + w1], gray[y2:y2 + h2, x2:x2 + w2]]
		crop_eyes_l = cv2.resize(crop_eyes[0], (100, 100))
		crop_eyes_l = np.array(crop_eyes_l)
		crop_eyes_l = crop_eyes_l.astype("float32")
		crop_eyes_l /= 255
		crop_eyes_r = cv2.resize(crop_eyes[1], (100, 100))
		crop_eyes_r = np.array(crop_eyes_r)
		crop_eyes_r = crop_eyes_r.astype("float32")
		crop_eyes_r /= 255
		return [crop_face.reshape((-1, 100, 100, 1)), crop_eyes_l.reshape((-1, 100, 100, 1)),
		        crop_eyes_r.reshape((-1, 100, 100, 1))]
	except Exception as e:
		return None


# def eye_cropper(frame):
# 	# create a variable for the facial feature coordinates
# 	facial_features_list = face_recognition.face_landmarks(frame)
#
# 	# create a placeholder list for the eye coordinates
# 	# and append coordinates for eyes to list unless eyes
# 	# weren't found by facial recognition
# 	try:
# 		eye = facial_features_list[0]['left_eye']
# 	except:
# 		try:
# 			eye = facial_features_list[0]['right_eye']
# 		except:
# 			return
#
# 	# establish the max x and y coordinates of the eye
# 	x_max = max([coordinate[0] for coordinate in eye])
# 	x_min = min([coordinate[0] for coordinate in eye])
# 	y_max = max([coordinate[1] for coordinate in eye])
# 	y_min = min([coordinate[1] for coordinate in eye])
#
# 	# establish the range of x and y coordinates
# 	x_range = x_max - x_min
# 	y_range = y_max - y_min
#
# 	# in order to make sure the full eye is captured,
# 	# calculate the coordinates of a square that has a
# 	# 50% cushion added to the axis with a larger range and
# 	# then match the smaller range to the cushioned larger range
# 	if x_range > y_range:
# 		right = round(.5 * x_range) + x_max
# 		left = x_min - round(.5 * x_range)
# 		bottom = round((((right - left) - y_range)) / 2) + y_max
# 		top = y_min - round((((right - left) - y_range)) / 2)
# 	else:
# 		bottom = round(.5 * y_range) + y_max
# 		top = y_min - round(.5 * y_range)
# 		right = round((((bottom - top) - x_range)) / 2) + x_max
# 		left = x_min - round((((bottom - top) - x_range)) / 2)
#
# 	# crop the image according to the coordinates determined above
# 	cropped = frame[top:(bottom + 1), left:(right + 1)]
#
# 	# resize the image
# 	cropped = cv2.resize(cropped, (80, 80))
# 	image_for_prediction = cropped.reshape(-1, 80, 80, 3)
#
# 	return image_for_prediction

def find_drowsy_vid(vid_name):
	# file_id = url.split("/")[-2]
	# destination = './input.mp4'
	# destination = './navya_drowsy.mp4'
	# download_file_from_google_drive(file_id, destination)
	destination = "./" + vid_name
	destination = str(destination)
	print(destination)

	cap = cv2.VideoCapture(destination)

	if not cap.isOpened():
		print("Error opening video file")

	results = []

	start_time = time.time()
	while cap.isOpened():
		ret, frame = cap.read()
		frame = cv2.flip(frame, 0)
		if ret == True:
			cv2.imshow("frame", frame)
			frame_imgs = get_cropped_image(frame)
			if frame_imgs is not None:
				end_time = time.time()
				if end_time - start_time < 4:
					continue
				yawn_prediction = yawn_model.predict(
					frame_imgs[0], batch_size=1)

				# eye1_prediction = eye_model.predict(frame_imgs)
				# print(eye1_prediction)
				eye1_prediction = eye_model.predict(
					frame_imgs[1], batch_size=1)
				eye2_prediction = eye_model.predict(
					frame_imgs[2], batch_size=1)
				if yawn_prediction[0][0] >= 0.5 or eye1_prediction[0][0] >= 0.5 and eye2_prediction[0][0] >= 0.5:
					mixer.music.play()
					results.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
				start_time = time.time()
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
		else:
			break

	cap.release()
	cv2.destroyAllWindows()
	os.remove(destination)
	return results
