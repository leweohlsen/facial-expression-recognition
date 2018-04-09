#!/usr/bin/env python

"""This module reads the CK+ dataset emotion labels and the corresponding
png images into a CSV file for easier processing in higher level apis. The
first image in each sequence is the neutral expression while the last image
is the peak expression.

All images are converted to 48x48 pixels grayscale.

To run this module successfully, you should have the same folder 
structure as shown in README.md.

    CK+ classes        FER classes         
    0=neutral     =>   6
    1=anger       =>   0  
    2=contempt    =>   maybe angry or disgust?
    3=disgust     =>   1
    4=fear        =>   2
    5=happy       =>   3
    6=sadness     =>   4
    7=surprise    =>   5

FER labels are ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
"""

import os
import glob
import cv2
import random

abspath = os.path.dirname(__file__)

# load OpenCV pre-trained cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(abspath + '/../model/haarcascade_frontalface_default.xml')

def map_class(ck_class):
    if ck_class == 0:
        return 6
    elif ck_class == 1:
        return 0
    elif ck_class == 2:
        return 0
    elif ck_class == 3:
        return 1
    elif ck_class == 4:
        return 2
    elif ck_class == 5:
        return 3
    elif ck_class == 6:
        return 4
    elif ck_class == 7:
        return 5

def detect_face(img):
    # find and extract faces on image (should be one face per image)
    # assuming all the falsely detected faces are smaller than the actual
    # face, so discard all location params except of the largest
    faces = face_cascade.detectMultiScale(img, 1.1, 5)
    (x, y, w, h) =  sorted(faces, key=len)[0]
    # print(x, y, w, h)
    return img[y:y+h, x:x+w]

# csv file to write to
csv_file = open('data/ckplus.csv', 'w')

# write csv header
csv_file.write("class,pixels\n")

for root, dirs, files in os.walk(abspath + '/../data/ckplus/labels'):
    for file in files:
        csv_line = ''
        label_path = os.path.join(root, file)
        # read labels
        with open(label_path, 'r') as f:
            label = map_class(int(float(f.read())))
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
        neutral_image = cv2.cvtColor(neutral_image, cv2.COLOR_BGR2GRAY)

        # detect faces
        peak_face = detect_face(peak_image)
        neutral_face = detect_face(neutral_image)

        # resize to 48x48 pixels -> better yet, 128x128 :)
        peak_face = cv2.resize(peak_face, (128, 128))
        neutral_face = cv2.resize(neutral_face, (128, 128))

        # cv2.imshow('Peak Face', peak_face)
        # cv2.waitKey()

        # cv2.imshow('Neutral Face', neutral_face)
        # cv2.waitKey()

        # write classes and image pixels to csv
        csv_line += " ".join(str(pixel) for pixel in peak_face.flatten())
        if (label != 2): # omit "contempt"-emotion for now 
            csv_file.write("%s\n" % csv_line)
        
        # we have much more neutral expressions than peak expressions,
        # so only every other neutral expression is taken for the csv
        # this is a temporary workaround instead of chossing the 
        # neutral samples manually
        if (random.random() < 1.0 / 4.0):
            csv_line = '6,'
            csv_line += " ".join(str(pixel) for pixel in neutral_face.flatten())
            csv_file.write("%s\n" % csv_line)

