import os
import sys
import random
import numpy as np
from tqdm import tqdm

import cv2
import matplotlib.pyplot as plt

ROOT_DIR = os.getcwd()
DATA_PATH = os.path.join(ROOT_DIR, "data")
TEST_IMGS_PATH = os.path.join(DATA_PATH, "images")
MODEL_PATH = os.path.join(ROOT_DIR, "model")

CV2_MODEL_PATH = os.path.join(MODEL_PATH, "cv2")
HAAR_WEIGHT_FILE = os.path.join(CV2_MODEL_PATH, "haarcascade_frontalface_default.xml")

facesDetector = cv2.CascadeClassifier(HAAR_WEIGHT_FILE)

TEST_VIDEOS_PATH = os.path.join(DATA_PATH, "videos")
video_inp =  os.path.join(TEST_VIDEOS_PATH, "test1080p_1.mp4")
video_out =  os.path.join(TEST_VIDEOS_PATH, "test1080p_1-haar.mp4")

video_reader = cv2.VideoCapture(video_inp)

nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

if int(major_ver)  < 3 :
    fps = video_reader.get(cv2.cv.CV_CAP_PROP_FPS)
    print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
else :
    fps = video_reader.get(cv2.CAP_PROP_FPS)
    print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
        
video_writer = cv2.VideoWriter(video_out,
                               cv2.VideoWriter_fourcc(*'XVID'), 
                               fps,
                               (frame_w, frame_h))

total_faces_detected = 0

for i in tqdm(range(nb_frames)):
    ret, bgr_image = video_reader.read()
    
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    
    faces = facesDetector.detectMultiScale(gray_image,1.3, 5)

    total_faces_detected += len(faces) 
    
    for (x, y, w, h) in faces:
       
        cv2.rectangle(bgr_image, (x,y), (x+w, y+h), (0,255,0), 2)
        
        #bgr_image[y+10:y+h-10,x:x+w,0]=np.random.normal(size=(h-20,w))
        #bgr_image[y+10:y+h-10,x:x+w,1]=np.random.normal(size=(h-20,w))
        #bgr_image[y+10:y+h-10,x:x+w,2]=np.random.normal(size=(h-20,w))
        
    video_writer.write(bgr_image)
    
video_reader.release()
video_writer.release()

print("Total faces detected: ", total_faces_detected)