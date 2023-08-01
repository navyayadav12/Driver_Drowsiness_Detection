import cv2
import requests
import os
import time
from tensorflow import keras
import numpy as np

from pygame import mixer

mixer.init()

mixer.music.load("emergency-alarm-with-reverb-29431.mp3")

mixer.music.set_volume(0.7)

yawn_model = keras.models.load_model("./yawn_model.h5",compile=False)
eye_model = keras.models.load_model("./aug_eye_model.h5",compile=False)

face = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('./haarcascades/haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('./haarcascades/haarcascade_righteye_2splits.xml')




lbl = ['Close', 'Open']
path = os.getcwd()
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count = 0
score = 0
thicc = 2
rpred = [99]
lpred = [99]
fpred = [99]

while True:
	ret, frame = cap.read()
	height, width = frame.shape[:2]
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	faces = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
	left_eye = leye.detectMultiScale(gray)
	right_eye = reye.detectMultiScale(gray)

	cv2.rectangle(frame, (0, height - 50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

	for (x, y, w, h) in faces:
		cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)
		fac = frame[y:y + h, x:x + w]
		count = count + 1
		fac = cv2.cvtColor(fac, cv2.COLOR_BGR2GRAY)
		fac = cv2.resize(fac, (100, 100))
		fac = fac / 255
		fac = fac.reshape(100, 100, -1)
		fac = np.expand_dims(fac, axis=0)
		fpred = yawn_model.predict(fac, batch_size=1)

	for (x, y, w, h) in right_eye:
		r_eye = frame[y:y + h, x:x + w]
		count = count + 1
		r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
		r_eye = cv2.resize(r_eye, (100, 100))
		r_eye = r_eye / 255
		r_eye = r_eye.reshape(100, 100, -1)
		r_eye = np.expand_dims(r_eye, axis=0)
		rpred = eye_model.predict(r_eye, batch_size=1)
		# print(lpred)
		# if lpred[0] >= 0.5:
		# 	lpred[0] = 1
		# else:
		# 	lpred[0] = 0
		if rpred[0] == 1:
			lbl = 'Closed'
		if rpred[0] == 0:
			lbl = 'Open'
		break

	for (x, y, w, h) in left_eye:
		l_eye = frame[y:y + h, x:x + w]
		count = count + 1
		l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
		l_eye = cv2.resize(l_eye, (100, 100))
		l_eye = l_eye / 255
		l_eye = l_eye.reshape(100, 100, -1)
		l_eye = np.expand_dims(l_eye, axis=0)
		lpred = eye_model.predict(l_eye, batch_size=1)
		# print(lpred[0])
		# temp = (int)lpred
		# if lpred[0] >= 0.5:
		# 	lpred[0] = 1
		# else:
		# 	lpred[0] = 0
		if lpred[0] == 1:
			lbl = 'Closed'
		if lpred[0] == 0:
			lbl = 'Open'
		break
	# print(lpred[0])
	# print(rpred[0])
	if fpred[0] >= 0.5 or rpred[0] >= 0.5 and lpred[0] >= 0.5:  # eye is close if val is 1
		score = score + 1
		cv2.putText(frame, "Closed", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
	# if(rpred[0]==1 or lpred[0]==1):
	else:
		score = score - 1
		cv2.putText(frame, "Open", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

	if score < 0:
		score = 0
	print(score)
	cv2.putText(frame, 'Score:' + str(score), (100, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
	if (score > 20):
		# person is feeling sleepy so we beep the alarm
		cv2.imwrite(os.path.join(path, 'image.jpg'), frame)
		try:
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
