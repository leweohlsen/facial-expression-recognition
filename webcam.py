#!/usr/bin/env python

import cv2
import numpy as np
import predict

# Get a reference to webcam #0 (the default one)
webcam = cv2.VideoCapture(0)

# load OpenCV pre-trained cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('model/haarcascade_frontalface_default.xml')

# variables
face_locations = []
face_emotions = []
frame_count = 0
process_every_n_frames = 10

while True:
    # Grab a single frame of the video
    ret, frame = webcam.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    frame_resized = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Only process every other frame of video to save time
    if frame_count % process_every_n_frames == 0:

        # convert image to grayscale
        frame_resized_gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

        # Find all the faces in the current frame of video
        faces = face_cascade.detectMultiScale(frame_resized_gray, 1.1, 5)

        print('Faces found: ', len(faces))

        face_emotions = []
        for (x, y, w, h) in faces:

            # print(x, y, w, h)

            face = frame_resized_gray[y:y+h, x:x+w]
            face_input_48 = cv2.resize(face, (48, 48)).reshape(1, 48*48)
            face_input_48 = np.array(face_input_48, dtype=np.float32)
            # face_input_48 = np.divide(face_input_48, 255)

            face_input_128 = cv2.resize(face, (128, 128)).reshape(1, 128*128)
            face_input_128 = np.array(face_input_128, dtype=np.float32)
            # face_input_128 = np.divide(face_input_128, 255)

            # pred_probas = predict.predictEmotion(face_input_intensity)
            # np.set_printoptions(precision=3, suppress=True)
            pred_class = predict.predictEmotion(face_input_128, face_input_48)
            print(pred_class)
            face_emotions.append(predict.class_labels[pred_class])


    # Display the results
    for (x, y, w, h), name in zip(faces, face_emotions):

        x *= 4
        y *= 4
        w*= 4
        h *= 4

        # Draw a box around the face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)

        # Draw a label with the classification result below the face
        cv2.rectangle(frame, (x, y+h), (x+w, y+h+30), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, ((x, y+h+25)), font, 0.8, (255, 255, 255), 1)

    frame_count += 1

    # Display the resulting image
    cv2.imshow('Facial Expression Recognition', frame)

    

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
webcam.release()
cv2.destroyAllWindows()
