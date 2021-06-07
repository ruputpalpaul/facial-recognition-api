# objective: Take the images in the data/currentImages folder and crop the faces and store it in data/cropedFaces folder

import time
import os
from os import listdir
from datetime import datetime
from os.path import isfile, join
import pickle
from predictStudent import PredictModel


def cropFaces(imagePath, cv2):

    path = './data/croppedFaces/'
    image = cv2.imread(
        "./data/currentImages/"+imagePath)

    # converting the image to greyScale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # initializing harcascade classifier
    faceCascade = cv2.CascadeClassifier(
        './constant/haarcascade_frontalface_default.xml')

    # detecting the face from the greyScale Image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(30, 30)
    )

    # printing out the number of faces found
    print("[INFO] Found {0} Faces.".format(len(faces)))

    # if faces are found cropping the face and storing it in a folder
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_color = image[y:y + h, x:x + w]
        print("[INFO] Object found. Saving locally.")

        # saving cropped images
        cv2.imwrite(os.path.join(path, str(w) +
                                 str(h) + '_faces.jpg'), roi_color)

    # if face is present storing the original image with the rectangles added to the faces
    if(len(faces) != 0):
        status = cv2.imwrite(os.path.join(
            './data/detectedImages/', str(time.time())+'.jpg'), image)
        print("[INFO] Image faces_detected.jpg written to filesystem: ", status)
    else:
        print("No face detected")

    if(len(faces) > 0):
        onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
        for i in onlyfiles:
            PredictModel(imagePath=path+'/'+i, cv2=cv2)
    
    return True

