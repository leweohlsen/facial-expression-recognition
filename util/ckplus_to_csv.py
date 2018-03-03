#!/usr/bin/env python

"""This module reads the CK+ dataset emotion labels and the corresponding
png images into a CSV file for easier processing in higher level apis.
All images are converted to 48x48 pixels grayscale.

To run this module, you should have the same folder structure as shown in 
README.md.

Labels are:
    0=neutral
    1=anger
    2=contempt
    3=disgust
    4=fear
    5=happy
    6=sadness
    7=surprise
"""

import os
import glob
import cv2

# csv file to write to
csv_file = open('../data/ckplus.csv', 'w')

for root, dirs, files in os.walk("../data/ckplus/labels"):
    for file in files:
        csv_line = ''
        label_path = os.path.join(root, file)
        # read labels
        with open(label_path, 'r') as f:
            label = int(float(f.read()))
            csv_line += str(label) + ','
            # print(label)
        # get images for label
        image_paths = root.replace('labels', 'images')
        image_paths = sorted(glob.glob(image_paths + '/*.png'))
        # last image is the peak expression
        peak_image = cv2.imread(image_paths[-1], flags=cv2.IMREAD_COLOR)
        neutral_image = cv2.imread(image_paths[0], flags=cv2.IMREAD_COLOR)

        # convert images to grayscale
        peak_image = cv2.cvtColor(peak_image, cv2.COLOR_BGR2GRAY)
        
        if (label == 2):
            cv2.imshow('Neutral Image', neutral_image)
            cv2.imshow('Peak Image', peak_image)
            cv2.waitKey()

        # load OpenCV pre-trained cascade classifier for face detection
        face_cascade = cv2.CascadeClassifier('../model/haarcascade_frontalface_default.xml')

        # find and extract faces on image (should be one face per image)
        # assuming all the falsely detected faces are smaller than the truly detected
        # face, so discard all location params except of the largest
        faces = face_cascade.detectMultiScale(peak_image, 1.1, 5)
        (x, y, w, h) =  sorted(faces, key=len)[0]
        # print(x, y, w, h)
        face = peak_image[y:y+h, x:x+w]
        # resize to 48x48 pixels
        face = cv2.resize(face, (48, 48))

        # write class and image pixels to csv
        csv_line += " ".join(str(pixel) for pixel in face.flatten())
        csv_file.write("%s\n" % csv_line)

        # cv2.imshow('Face', face)
        # cv2.waitKey()