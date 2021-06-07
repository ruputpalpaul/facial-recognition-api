from imageCapture import takeImage
from addStudentData import add_Student_Data
from train_Model import trainModel
from flask import Flask, render_template, request
import pandas as pd
import json
import os
import shutil


# initializing flask
app = Flask(__name__)


@app.route('/')
def index():
    return "This is api home page"


@app.route('/attendance/<ch>')
def home(ch):
    ch = int(ch)
    print("================ Automated Attendence system===================")
    print("1. Add new student")
    print("2. Train the system")
    print("3. Mark attendence")
    print("4. Exit the program.")
    print("===============================================================")
    # ch = int(input("Enter your choice: "))

    if ch == 1:
        add_Student_Data()
    elif ch == 2:
        trainModel()
    elif ch == 3:
        takeImage(cameraMode=0)
        df = pd.read_csv("Attendence_sheet.txt", delimiter=",")
        result = df.to_json(orient="split")
        parsed = json.loads(result)
        deleteCroppedImages()
        # clearAttendenceSheet()
        return json.dumps(parsed, indent=4)
    else:
        exit(0)

    return "attendence marked."


def deleteCroppedImages():

    folder = './data/croppedFaces'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def clearAttendenceSheet():
    f = open('Attendence_Sheet.txt', 'r+')
    f.truncate(0)
    f.close()


# entry point
if __name__ == '__main__':
    app.run(debug=True)
