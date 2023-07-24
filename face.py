import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
from keras.models import load_model
import numpy as np
import mediapipe as mp
from djitellopy import Tello
import time

# Initialize the Tello drone
tello = Tello()

# Connect to the Tello drone using the SDK
tello.connect()

# Add a delay after connecting to the drone
time.sleep(5)

# Takeoff
tello.takeoff()

# Enable video streaming
tello.streamon()

# Trained model
model = load_model('keras_model.h5')

# Function for class names
# class names found in label.txt
def get_className(classNo):
    if classNo == 1:
        return "a" - # label.txt
    elif classNo == 0:
        return "W" - # label.txt

# Mediapipe face detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Video capture  for laptop camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)  
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  
font = cv2.FONT_HERSHEY_COMPLEX

should_land = False

while True:
    # Read frame from laptop camera
    _, laptop_frame = cap.read()

    # Check if laptop_frame is empty
    if laptop_frame is None:
        continue

    laptop_frame_rgb = cv2.cvtColor(laptop_frame, cv2.COLOR_BGR2RGB)

    # Detect faces using Mediapipe on laptop frame
    laptop_results = face_detection.process(laptop_frame_rgb)

    if laptop_results.detections:
        for detection in laptop_results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = laptop_frame.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                    int(bboxC.width * iw), int(bboxC.height * ih)

            crop_img = laptop_frame[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
            if crop_img.size == 0:
                continue  

            img = cv2.resize(crop_img, (224, 224))
            img = img.reshape(1, 224, 224, 3)
            prediction = model.predict(img)
            classIndex = np.argmax(prediction)
            probabilityValue = np.amax(prediction)

            classLabel = get_className(classIndex)
            if probabilityValue < 0.5 or classLabel not in ["Patrick Omodara", "William Carpenter"]:
                classLabel = "Unrecognized Face"

            cv2.rectangle(laptop_frame, bbox, (0, 255, 0), 2)
            cv2.rectangle(laptop_frame, (bbox[0], bbox[1] - 40), (bbox[0] + bbox[2], bbox[1]), (0, 255, 0), -2)
            cv2.putText(laptop_frame, classLabel, (bbox[0], bbox[1] - 10), font, 0.75,
                        (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(laptop_frame, str(round(probabilityValue * 100, 2)) + "%", (180, 75), font, 0.75,
                        (255, 0, 0), 2, cv2.LINE_AA)

    # Display laptop camera frame
    cv2.imshow("Laptop Camera", laptop_frame)

    # Read frame from Tello video stream
    frame = tello.get_frame_read().frame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Detect faces using Mediapipe on Tello video stream frame
    tello_results = face_detection.process(frame_rgb)

    if tello_results.detections:
        for detection in tello_results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                    int(bboxC.width * iw), int(bboxC.height * ih)

            crop_img = frame[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
            if crop_img.size == 0:
                continue  # Skip empty crop_img

            img = cv2.resize(crop_img, (224, 224))
            img = img.reshape(1, 224, 224, 3)
            prediction = model.predict(img)
            classIndex = np.argmax(prediction)
            probabilityValue = np.amax(prediction)

            classLabel = get_className(classIndex)
            if probabilityValue < 0.5 or classLabel not in ["a", "W"]:
                classLabel = "Unrecognized Face"

            cv2.rectangle(frame, bbox, (0, 255, 0), 2)
            cv2.rectangle(frame, (bbox[0], bbox[1] - 40), (bbox[0] + bbox[2], bbox[1]), (0, 255, 0), -2)
            cv2.putText(frame, classLabel, (bbox[0], bbox[1] - 10), font, 0.75,
                        (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, str(round(probabilityValue * 100, 2)) + "%", (180, 75), font, 0.75,
                        (255, 0, 0), 2, cv2.LINE_AA)

    # Display Tello video stream frame
    cv2.imshow("Tello Video Stream", frame)

    # Check for 'q' key press
    k = cv2.waitKey(1)
    if k == ord('q'):
        should_land = True
        cv2.destroyAllWindows()
        break

# Land if 'q' was pressed
if should_land:
    # Add a delay before landing
    time.sleep(150)
    tello.land()

# Release resources
cap.release()

