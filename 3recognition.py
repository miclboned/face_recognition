#!/usr/bin/env python
#-*- coding:UTF-8 -*-
#file name:ch3-06/3recognition.py
import cv2
import numpy as np
#from keras.models import load_model
#model=load_model('face_recognition2.h5')
#model.summary()

model = cv2.face.LBPHFaceRecognizer_create()
model.read('faces.data')
print('load training data done')

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
cv2.namedWindow('video', cv2.WINDOW_NORMAL)
# 可識別化名稱
names = ['Mark','baby']

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (600, 336))
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#### 在while內
    faces = face_cascade.detectMultiScale(gray, 1.1, 3)
    for (x, y, w, h) in faces:
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        face_image = cv2.resize(gray[y: y + h, x: x + w], (400, 400))
        try:
            val = model.predict(face_image)
            print('label:{}, confidence:{}'.format(val[0], val[1]))
            if val[1] < 50:
                cv2.putText(
                    frame, names[val[0]], 
                    (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, 
                    (255,255,0), 3, cv2.LINE_AA
                )
        except:
            continue

#### 在while內
    cv2.imshow('video', frame)
    if cv2.waitKey(1) == 27:
        cv2.destroyAllWindows()
        break
