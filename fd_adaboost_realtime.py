import os
import cv2

ROOT_DIR = os.getcwd()
DATA_PATH = os.path.join(ROOT_DIR, "data")
TEST_IMGS_PATH = os.path.join(DATA_PATH, "images")
MODEL_PATH = os.path.join(ROOT_DIR, "model")

CV2_MODEL_PATH = os.path.join(MODEL_PATH, "cv2")
HAAR_WEIGHT_FILE = os.path.join(CV2_MODEL_PATH, "haarcascade_frontalface_default.xml")

facesDetector = cv2.CascadeClassifier(HAAR_WEIGHT_FILE)

video_capture = cv2.VideoCapture(0)

total_faces_detected = 0

while(video_capture.isOpened()):
    ret, bgr_image = video_capture.read()

    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    
    faces = facesDetector.detectMultiScale(gray_image, 1.3, 5)

    total_faces_detected += len(faces) 
    
    for (x, y, w, h) in faces:
        cv2.rectangle(bgr_image, (x,y), (x+w, y+h), (0,255,0), 2)

    cv2.imshow('Video', bgr_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()

print("Total faces detected: ", total_faces_detected)