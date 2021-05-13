# -*- coding: utf-8 -*-
"""
Created on Wed May 12 11:02:22 2021

@author: Karol
"""

import dlib
import cv2
import imutils
import os
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
from azure.storage.blob import ContainerClient

#選擇第一支攝影機
cap = cv2.VideoCapture(0)
cap.set(cv2. CAP_PROP_FRAME_WIDTH, 650)
cap.set(cv2. CAP_PROP_FRAME_HEIGHT, 500)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(".\\shape_predictor_68_face_landmarks.dat\\shape_predictor_68_face_landmarks.dat")

while (cap.isOpened()):

    ret, frame =cap.read()
    face_rects, scores, idx = detector.run(frame, 0)

    for i, d in enumerate(face_rects):
        x1 = d.left()
        y1 = d.top()
        x2 = d.right()
        y2 = d.bottom()
        text = " %2.2f (%d)" % (scores[i] , idx[i])
        
        #繪製偵測人臉圖的矩形範圍
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 4, cv2.LINE_AA)       
        cv2.putText(frame, text, (x1,y1), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255,255,255),1 ,cv2.LINE_AA)
       
        landmarks_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)       
        shape = predictor(landmarks_frame, d)
        
        #繪製68個特徵點 #紅點就是偵測出人臉的68個特徵點
        for i in range(68):
            cv2.circle(frame, (shape.part(i).x, shape.part(i).y), 3, (0, 0, 255), 2)
            cv2.putText(frame, str(i), (shape.part(i).x, shape.part(i).y ), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0 , 0), 1)
        
    cv2.imshow("Face Detection", frame)
    
    if cv2.waitKey(1) & 0xFF ==ord('q'): #按下q，截圖儲存並退出。
        cv2.imwrite(".\\001.jpg", frame)
        break
        
        #如果按下ESC就退出
    if cv2.waitKey(10) == 27:
        break

cap.release()
cv2.destroyAllWindows()

img = cv2.imread('.\\001.jpg')
img_array = np.array(img)
#type(img)

image_stream = BytesIO()
plt.savefig(image_stream)
image_stream.seek(0)

# upload in blob storage
container_client = ContainerClient.from_container_url("your_container_SASconnection_string")
blob_client = container_client.get_blob_client(blob ="001.jpg")
blob_client.upload_blob(image_stream.read(), blob_type="BlockBlob")
