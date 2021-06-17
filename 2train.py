#!/usr/bin/env python
#-*- coding:UTF-8 -*-
#file name:ch3-06/2train.py
import cv2
import numpy as np

images = []
labels = []
# 第一張人臉的標籤為0
for index in range(100):
    filename = 'images/h0/{:02d}.pgm'.format(index)
    print('read ' + filename)
    img = cv2.imread(filename, cv2.COLOR_BGR2GRAY)
    images.append(img)
    labels.append(0)
# 第二張人臉的標籤為1
for index in range(100):
    filename = 'images/h1/{:02d}.pgm'.format(index)
    print('read ' + filename)
    img = cv2.imread(filename, cv2.COLOR_BGR2GRAY)
    images.append(img)
    labels.append(1)

print('training...')
#model = cv2.createLBPHFaceRecognizer()
model = cv2.face.LBPHFaceRecognizer_create()
model.train(np.asarray(images), np.asarray(labels))
model.save('faces.data')
print('training done')
