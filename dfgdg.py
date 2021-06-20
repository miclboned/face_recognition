# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 14:33:32 2021

@author: Wu
"""

import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt
index=0
#face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')


#filename="C:/Users/ZhiXuan/Desktop/Matt LeBlanc/Matt LeBlanc_"+str(index)+".jpg"
#path="C:/Users/ZhiXuan/Desktop/train/Matt LeBlanc/Matt LeBlanc_"+str(index)+".jpg"


for index in range(80):
    
    filename="C:/Users/ZhiXuan/Desktop/David Schwimmer​/David Schwimmer ​_"+str(index)+".jpg"
    img =cv2.imread(filename)
    imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faceCascade=cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    faces=faceCascade.detectMultiScale(imgGray,1.1,4)   

    for(x,y,w,h)in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        face_image = cv2.resize(imgGray[y: y + h, x: x + w], (200, 200))
        print(len(faces))

        cv2.imwrite("C:/Users/ZhiXuan/Desktop/train/David Schwimmer​​/David Schwimmer ​_"+str(index)+".jpg", face_image)
   # cv2.imshow("Result2",face_image)
        cv2.imshow("Result",img)
        cv2.waitKey(0)==13
#def extractFace(srcpath, dstpath):
#    if not os.path.exists(srcpath):
#       os.mkdir(srcpath)
#    if not os.path.exists(dstpath):
#       os.mkdir(dstpath)
#    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
#    for fname in os.listdir(srcpath):
#        img = Image.open(srcpath + fname)
#        imgary = cv2.imread(srcpath + fname)
#        faces = face_cascade.detectMultiScale(imgary, 1.3, 5)
        # if len(faces) == 1:
#        x,y,w,h = faces[0]
#        crpim = img.crop((x,y, x + w, y + h)).resize((64,64))
#        crpim.save(dstpath + fname)
  

#srcpath = 'C:/Users/ZhiXuan/Desktop/Matt LeBlanc​/' 
#dstpath = 'C:/Users/ZhiXuan/Desktop/train/Matt LeBlanc​/'
#extractFace(srcpath, dstpath)
