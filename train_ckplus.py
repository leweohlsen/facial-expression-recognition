#!/usr/bin/env python

from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import random
import model
import csv

# Training Parameters
num_steps = 250
batch_size = 16

# images and labels
imgs = []
labels = []

# read csv file
with open('data/ckplus.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        pixels_test = [float(x) for x in row[1].split(' ')]
        imgs.append(np.array(pixels_test))
        labels.append(int(row[0]))

# wrap into np arrays and divide pixels by 255
imgs = np.array(imgs, dtype=np.float32)
imgs = np.divide(imgs, 255)
labels = np.array(labels, dtype=np.float32)

# TODO: "leave one out" cross validation?

# use sklearn for dividing dataset randomly (pseudo-random for now..)
random.seed(42)
imgs_train, imgs_test, labels_train, labels_test = train_test_split(imgs, labels, test_size=0.2, random_state=42)

# Build the Estimator
estimator = tf.estimator.Estimator(
    model_fn=model.model_fn, 
    model_dir='./model/ckplus',
    params={
        'learning_rate': 0.001,
        'num_classes': 7,
        'img_size': 128,
        'dropout_rate': 0.25
    })

# Training
# Define the input function for training
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': imgs_train}, y=labels_train,
    batch_size=batch_size, num_epochs=None, shuffle=True)
# Train the Model
estimator.train(input_fn, steps=num_steps)



# Evaluate the Model
# Define the input function for evaluating
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': imgs_test}, y=labels_test,
    batch_size=batch_size, shuffle=False)
# Use the Estimator 'evaluate' method
estimator.evaluate(input_fn)