import cv2
from deepface import DeepFace
from GazeTracking.gaze_tracking import GazeTracking
gaze = GazeTracking()


import numpy as np
# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start capturing video
cap = cv2.VideoCapture(0)


# Define the dimensions of the pad
pad_width = 200
pad_height = 480

# for cv2 put text
font = cv2.FONT_HERSHEY_PLAIN
fontScale = 1
color = (0, 255, 0)
thickness = 1


while True:
    # Capture frame-by-frame
    _, frame = cap.read()


    # Create a pad with the specified dimensions
    left_pad = np.ones((pad_height, pad_width, 3), dtype=np.uint8)
    
    # flip the frame
    frame = cv2.flip(frame, flipCode=1)
    frame = np.hstack([left_pad, frame])


    # We send this frame to GazeTracking to analyze it
    gaze.refresh(frame)

    frame = gaze.annotated_frame()
    gaze_text = ""

    if gaze.is_blinking():
        gaze_text = "blink"
    elif gaze.is_right():
        gaze_text = "right"
    elif gaze.is_left():
        gaze_text = "left"
    elif gaze.is_center():
        gaze_text = "center"

    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    frame = cv2.putText(frame, f"Gaze: {gaze_text}", (5, 20), font, fontScale, color, thickness)
    frame = cv2.putText(frame, f"x coor: {left_pupil}", (5, 40), font, fontScale, color, thickness)
    frame = cv2.putText(frame, f"y coor: {right_pupil}", (5, 60), font, fontScale, color, thickness)

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convert grayscale frame to RGB format
    rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the face ROI (Region of Interest)
        face_roi = rgb_frame[y:y + h, x:x + w]

        
        # Perform emotion analysis on the face ROI
        result = DeepFace.analyze(face_roi, actions=['age', 'gender', 'race', 'emotion'], enforce_detection=False)

        # Determine the dominant emotion
        emotion = result[0]['dominant_emotion']
        race = result[0]['dominant_race']
        gender = result[0]['dominant_gender']
        age = result[0]['age']

        # Draw rectangle around face and label with predicted emotion
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        # cv2.putText(frame, race, (x, y - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        # cv2.putText(frame, gender, (x, y - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        # cv2.putText(frame, str(age), (x, y - 75), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        frame = cv2.putText(frame, f"age: {age}", (5, 100), font, fontScale, color, thickness)
        frame = cv2.putText(frame, f"gender: {gender}", (5, 120), font, fontScale, color, thickness)
        frame = cv2.putText(frame, f"race: {race}", (5, 140), font, fontScale, color, thickness)
        frame = cv2.putText(frame, f"emotion: {emotion}", (5, 160), font, fontScale, color, thickness)

    # Display the resulting frame
    cv2.imshow('Real-time Emotion Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()

