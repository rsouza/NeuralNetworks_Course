import cv2
import sys
import logging as log
import datetime as dt
from time import sleep
import time
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from os import getcwd
from tensorflow.keras.models import load_model
import shutil


path_folder_imgs = "experiments/image-testing"
path_imgs = "experiments/image-testing/images"

model_loaded = load_model("model-tf-keras-covid-detection.h5")

def delete_files_directory(path_imgs):
    filepath = ""
    for filename in os.listdir(path_imgs):
        print(filename)
        filepath = os.path.join(path_imgs, filename)
    print(filepath)
    shutil.rmtree(path_imgs)
        
def create_directory(path_imgs,path_folder_imgs):
    delete_files_directory(path_imgs)
    try:
        os.mkdir(path_folder_imgs)
    except:
        error = "Directory already exists"
    try:
        os.mkdir(path_imgs)
    except:
        error = "Directory already exists"


cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
log.basicConfig(filename='webcam.log',level=log.INFO)

video_capture = cv2.VideoCapture(0)
anterior = 0

while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    create_directory(path_imgs,path_folder_imgs)
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    font = cv2.FONT_HERSHEY_SIMPLEX 

    FaceFileName = "/home/lexlabs/experiments/image-testing/images/face_"+"-"+str(dt.datetime.now()) + ".jpg"
    cv2.imwrite(FaceFileName, frame)
    

    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        "/home/lexlabs/experiments/image-testing",
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary',
        shuffle=False) 
    if(len(test_generator)>0):
       pred=model_loaded.predict(test_generator, steps=len(test_generator), verbose=1) 

    predicted_prob = pred[0][0]
    print(pred)
    if(predicted_prob<0.5):
        predicted = "COM MASCARA - UFA!"
    else:
        predicted = "SEM MASCARA - CORRE!!!"

    cv2.putText(frame,  
                    predicted,  
                    (50, 50),  
                    font, 1,  
                    (0, 255, 255),  
                    2,  
                    cv2.LINE_4)     

    for (x, y, w, h) in faces:
        cv2.putText(frame,  
                    predicted,  
                    (x, y),  
                    font, 1,  
                    (0, 255, 255),  
                    2,  
                    cv2.LINE_4) 
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)


    if anterior != len(faces):
        anterior = len(faces)
        log.info("faces: "+str(len(faces))+" at "+str(dt.datetime.now()))


    # Display the resulting frame
    cv2.imshow('Video', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Display the resulting frame
    cv2.imshow('Video', frame)

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()