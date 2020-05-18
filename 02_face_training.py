#this file is use to train the images captured by the 01_face_dataset.py
#after execution of this progtam triner.yml will be created

import cv2
import numpy as np
from PIL import Image
import os

# Path for face image database
path = 'dataset'

#it is use to classify the faces 
recognizer = cv2.face.LBPHFaceRecognizer_create()
#open cv's predefined classifire
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

# function to get the images and label data
def getImagesAndLabels(path):
    # path of the image
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples=[]
    ids = []

    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L') # convert it to gray color

        img_numpy = np.array(PIL_img,'uint8')
        #for collecting the faces having same id
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)
        #starting points of face and height & width 
        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)

    return faceSamples,ids

print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces,ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))

# Save the model into trainer/trainer.yml
recognizer.write('trainer/trainer.yml')

# Print the numer of faces trained and end program
print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))