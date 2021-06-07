import os
from PIL import Image
import numpy as np
import cv2
import pickle


def trainModel():
    face_cascade = cv2.CascadeClassifier(
        "./constant/haarcascade_frontalface_alt2.xml")
    eye_cascade = cv2.CascadeClassifier(
        "./constant/haarcascade_eye.xml")

    recognizer = cv2.face.LBPHFaceRecognizer_create()

    
    image_dir = "./data/studentFaces"

    current_id = 0
    label_ids = {}
    x_train = []
    y_labels = []

    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith("png") or file.endswith("jpg"):
                path = os.path.join(root, file)
                label = os.path.basename(os.path.dirname(
                    path)).replace(" ", "-").lower()
                print(label, path)
                if not label in label_ids:
                    #Creating the dictionary to save the name of the student with a number
                    label_ids[label] = current_id
                    current_id += 1

                id_ = label_ids[label]
                print(label_ids)
                # x_train.append(path) #verify the image and turn into a NUMPY array,GRAY
                # y_labels.append(label)
                pil_image = Image.open(path).convert("L")
                # size = (550, 550)  #get good data and skip resize
                # final_image = pil_image.resize(size, Image.ANTIALIAS)
                image_array = np.array(pil_image, "uint8")
                # print(image_array)
                faces = face_cascade.detectMultiScale(
                    image_array, scaleFactor=1.5, minNeighbors=5)
                for (x, y, w, h) in faces:
                    roi = image_array[y:y+h, x:x+w]
                    x_train.append(roi)
                    y_labels.append(id_)

    # print(y_labels)
    # print(x_train)
    with open("labels.pickle", "wb") as f:
        pickle.dump(label_ids, f)

    #Training The model for face detection
    recognizer.train(x_train, np.array(y_labels))
    #Saving the trained model to use at later stage.
    recognizer.save("trainner.yml")
    print("YOUR FACE CAN BE RECOGNIZED NEXT TIME.")
    #print(y_labels)
