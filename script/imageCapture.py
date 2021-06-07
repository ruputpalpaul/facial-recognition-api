# objective: Capture 3 image from the camera and store it in data/currentImage folder

from imageCropper import cropFaces

# importing the necessary packages
import cv2
import time
import os

# function to take image using the camera cameraMode=0 for laptop cam and cameraMode=1 for external webcam


def takeImage(cameraMode):
    path = "./data/currentImages/"
    camera = cv2.VideoCapture(cameraMode)
    return_value, image = camera.read()
    ts = str(time.time())
    fileName = ts+'.png'
    cv2.imwrite(os.path.join(path, fileName), image)
    print("Image Captured")
    camera.release()
    del(camera)
    cropFaces(imagePath=fileName, cv2=cv2)
    return True
