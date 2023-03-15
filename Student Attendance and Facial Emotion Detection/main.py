# Importing Libraries
import face_recognition
import os
import time
from datetime import datetime
import numpy as np
import cv2
import pandas as pd
from keras.models import load_model
import warnings
warnings.filterwarnings('ignore')


# Accessing camera and models and defining variables
start_time = time.time()
video = cv2.VideoCapture(0)
model = load_model('model_file.h5')
ret, frame = video.read()

# Defining HaarCascade for face detecting
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default_t1.xml')

#Definingn emotion array and variables
labels_dict = {0:'Angry',1:'Disgust', 2:'Fear', 3:'Happy',4:'Neutral',5:'Sad',6:'Surprise'}
data = []
path = 'Student_Photo_Database'
images = []
classNames = []
myList = os.listdir(path)


#Accessing student name through photo_name
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])


# Feature Extraction
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList



# Creating function for attendance
def markAttendance(name):
    with open('Attendance_Sheet.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            tString = now.strftime('%H:%M:%S')
            dtString = now.strftime("%Y-%m-%d")
            f.writelines(f'\n{name},{dtString},{tString}')



# Real Time Student Face detection and recognition
encodeListKnown = findEncodings(images)
while True:
    success, img = video.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS= cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)


    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)

    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].capitalize()
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
            markAttendance(name)

    cv2.imshow('Student Attendance',img)
    cv2.waitKey(1)



    # Check if 5 minutes have elapsed (currenly set to 0 for demonstration)
    if time.time() - start_time >= 0:
        start_time = time.time()

        # Real time Emotion Detection
        ret, frame = video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceDetect.detectMultiScale(gray, 1.3, 3)
        for x, y, w, h in faces:
            sub_face_img = gray[y:y + h, x:x + w]
            resized = cv2.resize(sub_face_img, (48, 48))
            normalize = resized / 255.0
            reshaped = np.reshape(normalize, (1, 48, 48, 1))
            result = model.predict(reshaped)
            label = np.argmax(result, axis=1)[0]
            print(labels_dict[label])
            data.append(labels_dict[label])

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y - 40), (x + w, y), (0, 255, 0), -1)
            cv2.putText(frame, labels_dict[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Save the DataFrame to an Excel file
            df = pd.DataFrame({'Observations': data})
            df.to_excel('Facial_Emototion_data.xlsx', sheet_name='Student Emotion Data', index=False)

        #Showing output in Emotion Frame
        cv2.imshow("Student Emotion", frame)
        cv2.waitKey(1)

    # Wait for 1 millisecond before checking the time again
    cv2.waitKey(1)


# Release the video and destroy all windows
video.release()
cv2.destroyAllWindows()




