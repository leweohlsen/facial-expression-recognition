import face_recognition
import cv2
import numpy as np
import predict

# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Initialize some variables
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
face_locations = []
face_encodings = []
face_emotions = []
frame_count = 0
process_every_n_frames = 10

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if frame_count % process_every_n_frames == 0:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)

        face_emotions = []
        for (top, right, bottom, left) in face_locations:
            # Sacle face locations back up
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            # See if we can find out the facial expression

            face_image_rgb = frame[top:bottom, left:right]
            face_image_gs = cv2.cvtColor(face_image_rgb, cv2.COLOR_BGR2GRAY)
            face_image_gs_resized = cv2.resize(face_image_gs, (48, 48)) 
            

            img_byte = np.array(face_image_gs_resized, dtype=np.float32)
            img = np.divide(img_byte, 255)

            pred_class = predict.predictEmotion(img)

            face_emotions.append(class_labels[pred_class])


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_emotions):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    frame_count += 1

    # Display the resulting image
    cv2.imshow('Video', frame)

    

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
