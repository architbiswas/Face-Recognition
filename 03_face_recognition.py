# this is the last file use to recognize the image

import cv2
import numpy as np
import os 
import sqlite3
#collect all the generated data in veriables
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
# it will set the font style
font = cv2.FONT_HERSHEY_SIMPLEX

#iniciate id counter
id = 1
# thos function will return the id and name.
def getProfile(id):
    conn = sqlite3.connect('Facebase.db')
    cmd = "SELECT * FROM People WHERE ID="+str(id)
    cursor = conn.execute(cmd)
    profile = None
    for row in cursor:
        profile = row
    conn.close()
    return profile

# Initialize and start realtime video capture
# when you are using drone camera then change 0 to 1
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video widht
cam.set(4, 480) # set video height

# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

while True:
# same as 01_face_data.py
    ret, img =cam.read()

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )

    for(x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        # this is perdefined function of opencv which will predict the faces
        # confidence is the % of match 
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        names = getProfile(id)

        if (confidence < 100):
            id = names
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            
            id = "Unknown"
            confidence = "  {0}%".format(round(100 - confidence))
        #to peint name id and % on image
        cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  
    
    cv2.imshow('camera',img) 

    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()