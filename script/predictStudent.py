import time
import os
from os import listdir
from datetime import datetime
from os.path import isfile, join
import pickle


def PredictModel(imagePath, cv2):

    labels = {}
    with open("labels.pickle", 'rb') as f:
        og_labels = pickle.load(f)
        labels = {v: k for k, v in og_labels.items()}

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("trainner.yml")
    now = datetime.now()
    file1 = open("Attendence_Sheet.txt", "a")
    f = 0
    while(True):
        image = cv2.imread(imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        id_, conf = recognizer.predict(gray)
        if conf < 50:
            # ! Need to be improved all the not predicted image has to be send to the app for manual attendence
            print("Image not recognized==={}".format(imagePath))
            continue
        if conf > 60:
            name = labels[id_]
            currentStamp = now.strftime("%m/%d/%Y, %H:%M:%S")
            attendence = "\n"+name+","+str(currentStamp)
            file1.write(attendence)
            print("[INFO] Attendence for "+name+" is marked.")

            path = os.path.join(
                './data/studentFaces/{}/'.format(name.replace("-", " ")), str(time.time())+'.png')

            status = cv2.imwrite(path, image)
            print("[INFO] Image faces_detected.jpg written to filesystem: ", status)
            break
    return True
