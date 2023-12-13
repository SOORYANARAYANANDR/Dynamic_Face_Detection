# -*- coding: utf-8
#negative square
"""list"""
import numpy as np
#importing face_recognition module to detect faces

#face_recognition is a dlib based module
# and dlib helps to encode  face with 128 different features
import face_recognition
#importing opencv and os to read images
import os
import cv2

from datetime import datetime

#give paths of images
path = 'images'
#creating list called images
images = []
personName = []
myList = os.listdir(path)
print("list of iamge names = ",myList,"\n")
for cu_img in myList:
    current_Img = cv2.imread(f'{path}/{cu_img}')
    images.append(current_Img)
    personName.append(os.path.splitext(cu_img)[0])

print("\twithout extension = ",personName)

#by using dlib we are enconding facial points
def faceEncodings(images):
    encodeList = []
    for img in images:
        #cv2 shows images in BGR format 
        #so we convert it to RGB format
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

print("128 different faceencodings for each face = \n",faceEncodings(images))
encodeListKnown = faceEncodings(images)
print("All encondings completed..!")


#giving access of camera to capture faces
cap = cv2.VideoCapture(0)

while True:
    #to read captured frames
    ret, frame = cap.read()
    #to rsize the pixels of frames captured
    faces = cv2.resize(frame, (0,0), None, 0.25, 0.25)
    #now again convert frames from bgr to rgb
    faces = cv2.cvtColor(faces, cv2.COLOR_BGR2RGB)

    #to find_location and face_encodings on the face captured by camera
    facesCurrentFrame = face_recognition.face_locations(faces)
    encodesCurrentFrame = face_recognition.face_encodings(faces, facesCurrentFrame)

    for encodeFace, faceLoc in zip(encodesCurrentFrame, facesCurrentFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        
        #to find minimum distance between face
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = personName[matchIndex].upper()
            print(name)
            y1,x2,y2,x1 = faceLoc
            y1,x2,y2,x1 = y1*4 ,x2*4 ,y2*4,x1*4
            #drawing rectangle
            # first defining image called frame , 
            # then point1 i.e (x1,y1) 
            # then point2 i.e (x2,y2) 
            # then def color i.e (0,255,0)green 
            # then def width i.e 3 
            cv2.rectangle(frame, (x1,y1),(x2,y2), (0,255,0),3)
        else:
            print("no match found")

    cv2.imshow("camera", frame)
    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()