# This file will capture the 300 images of the person in front of camera
# while executing this file only one person has to be in fornt of camera
# always insert the new number for new face
# write thw name in double quotes Ex: "ABC"

import cv2
import os
import sqlite3

#this will capture the video
# when you are using drone camera then change 0 to 1
cam = cv2.VideoCapture(0)#(1)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height

#to scan the faces
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#thus function is used to update the database.
def insertOrUpdate(Id,Name):
    #establish the connection with database 
    conn = sqlite3.connect("Facebase.db")
    # select all values from database having given id
    cmd = "SELECT * FROM People WHERE ID="+str(Id)
    #courser is the iterator which is act like 'i' in for loop
    cursor = conn.execute(cmd)
    isRecordExist = 0
    for row in cursor:
        isRecordExist = 1
    if(isRecordExist == 1):
        cmd = "UPDATE People SET Names="+str(Name)+" WHERE ID="+str(Id)
    else:
        # sql query to insert the name and id
        cmd = "INSERT INTO People(ID,Names) VALUES("+str(Id)+","+str(Name)+")"
    conn.execute(cmd)
    # it will save the changes in database
    conn.commit()
    # remove the connection
    conn.close()


# For each person, enter one numeric face id
face_id = input('\n enter user id :\t')
name = input('Enter Name:\t')
# calling of function to store the vales in database
insertOrUpdate(face_id,name)

print("\n [INFO] Initializing face capture. Look the camera and wait ...")
# Initialize individual sampling face count
count = 0

while(True):

    ret, img = cam.read()
    # image will caoture in gray color
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    
    #starting points of face and height & width 
    for (x,y,w,h) in faces:
        #it will draw a rectangle to face
        #color will be in format of (r,g,b)---+this(red,green,blue)
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
        count += 1

        # Save the captured image into the datasets folder
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
        # dispkay the image
        cv2.imshow('image', img)

    k = cv2.waitKey(30) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
    elif count >= 300: # Take 300 face sample and stop video
         break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
# it will close the camera and relese the control
cam.release()
# it eill close all the windows like window which shows the video
cv2.destroyAllWindows()
