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
faces_preds = []
frame_count = 0
process_every_n_frames = 20

# colors
white = (255, 255, 255)
black = (0, 0, 0)
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
emotion_colors = [(54, 67, 244), (136, 150, 0), (76, 39, 156), (80, 175, 76), (0, 152, 255), (59, 235, 255), (243, 150, 33)]


while True:
    # Grab a single frame of the video
    ret, frame = webcam.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    frame_resized = frame # cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Only process every other frame of video to save time
    if frame_count % process_every_n_frames == 0:

        # convert image to grayscale
        frame_resized_gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

        # Find all the faces in the current frame of video
        faces = face_cascade.detectMultiScale(frame_resized_gray, 1.1, 5)

        print('Faces found: ', len(faces))

        faces_preds = []
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
            prediction = predict.predict_emotion_probas(face_input_128, face_input_48)
            print(prediction)
            faces_preds.append(prediction)


    # Display the results
    for (x, y, w, h), (classes, probas) in zip(faces, faces_preds):

        # x *= 4
        # y *= 4
        # w *= 4
        # h *= 4

        # Draw a box around the face
        cv2.rectangle(frame,(x,y),(x+w,y+h), white, 2)

        # Draw a label with the classification result below the face
        cv2.rectangle(frame, (x-1, y+h), (x+w+1, y+h+70), white, cv2.FILLED)

        # prepare strings for the top 3 predicted class labels and probabilities
        top1 = predict.class_labels[classes[0]] + ' ' + str(int(probas[classes[0]] * 100)) + '%'
        top2 = predict.class_labels[classes[1]] + ' ' + str(int(probas[classes[1]] * 100)) + '%'
        top3 = predict.class_labels[classes[2]] + ' ' + str(int(probas[classes[2]] * 100)) + '%'

        # probability bars
        cv2.rectangle(frame, (x+1, y+h), (x+int(w * probas[classes[0]]), y+h+20), emotion_colors[classes[0]], cv2.FILLED)
        cv2.rectangle(frame, (x+1, y+h+20), (x+int(w * probas[classes[1]]), y+h+40), emotion_colors[classes[1]], cv2.FILLED)
        cv2.rectangle(frame, (x+1, y+h+40), (x+int(w * probas[classes[2]]), y+h+60), emotion_colors[classes[2]], cv2.FILLED)

        # emotion labels
        font = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(frame, top1, ((x, y+h+18)), font, 1.4, black, 2)
        cv2.putText(frame, top2, ((x, y+h+38)), font, 1.4, black, 1)
        cv2.putText(frame, top3, ((x, y+h+58)), font, 1.4, black, 1)


    frame_count += 1

    # Display the resulting image
    cv2.imshow('Facial Expression Recognition', frame)

    

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
webcam.release()
cv2.destroyAllWindows()
