#!/usr/bin/env python
#-*- coding:UTF-8 -*-
#file name:ch3-06/1capture.py
import cv2
ESC = 27
# 畫面數量計數
n = 1
# 存檔檔名用
index = 0
# 人臉取樣總數
total = 100

#存檔目錄
def saveImage(face_image, index):
    filename = "/home/pi/face_recognition/dataset/train/"+"WuZhiXaun."+str(index)+".jpg"
    filename2 = "/home/pi/face_recognition/dataset/test/"+"WuZhiXaun."+str(index)+".jpg"
    filename3 = "/home/pi/face_recognition/dataset/vaild/"+"WuZhiXaun."+str(index)+".jpg"
    cv2.imwrite(filename, face_image)
    cv2.imwrite(filename2, face_image)
    cv2.imwrite(filename3, face_image)
    print(filename)
    print(filename2)
    print(filename3)
    
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
cv2.namedWindow('video', cv2.WINDOW_NORMAL)

while n > 0:
    ret, frame = cap.read()
    #frame = cv2.resize(frame, (600, 336))
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#### 在while內,有偵測到人臉做存檔
    faces = face_cascade.detectMultiScale(gray, 1.1, 3)
    for (x, y, w, h) in faces:
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        if n % 5 == 0:
            face_image = cv2.resize(gray[y: y + h, x: x + w], (200, 200))
            saveImage(face_image, index)
	    #cv2.putText(face_image,str(index),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            index += 1
            if index >= total:
                print('get training data done')
                n = -1
                break
        n += 1

#### 在while內
    cv2.imshow('video', frame)
    if cv2.waitKey(1) == 27:
        cv2.destroyAllWindows()
        break
