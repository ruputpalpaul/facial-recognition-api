import cv2
import sys
import os
import time
from os import listdir
from os.path import isfile, join
import numpy as np
import pickle


def add_Student_Data():

    # initializing harcascade classifier
    faceCascade = cv2.CascadeClassifier(
        './constant/haarcascade_frontalface_default.xml')

    side_face_cascade = cv2.CascadeClassifier(
        './constant/haarcascade_profileface.xml')

    # starting video capture
    cap = cv2.VideoCapture(0)

    name = input("enter your name: ")
    name=name.lower()
    image_dir = "./data/studentFaces/"+name+"/"
    # creating a folder for each new student
    os.mkdir(image_dir)

    f = 0

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray, scaleFactor=1.5, minNeighbors=5)
        profileface = side_face_cascade.detectMultiScale(
            gray, scaleFactor=1.5, minNeighbors=5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]  # (ycord_start, ycord_end)
            roi_color = frame[y:y+h, x:x+w]

            img_item = "image{}.png".format(f)
            f += 1
            if f % 10 == 0:
                print(img_item)
            # storing the image into the folder
            cv2.imwrite(os.path.join(image_dir, img_item), frame)

            # adding rectangle to the face detected
            color = (255, 0, 0)
            stroke = 2
            end_cord_x = x + w
            end_cord_y = y + h
            cv2.rectangle(frame, (x, y), (end_cord_x,
                                          end_cord_y), color, stroke)

        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

        # taking 200 photos and then closing the video feed
        if f == 200:
            cap.release()
            cv2.destroyAllWindows()
            break

    print("Image successfully added to the data base.")
    cap.release()
    cv2.destroyAllWindows()

