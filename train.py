import numpy as np
import tensorflow as tf
import csv

import model

# Training Parameters
num_steps = 20000
batch_size = 128

# Feature Vectors (0-255)
imgs_train_byte = []
imgs_test_byte = []

# Labels (0-6)
labels_train_class = []
labels_test_class = []


with open('data/fer2013.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    # skip CSV header
    next(readCSV)
    for row in readCSV:
        if row[2] == 'Training':
            # cast pixels to int
            pixels_train = [float(x) for x in row[1].split(' ')]
            imgs_train_byte.append(np.array(pixels_train))
            labels_train_class.append(int(row[0]))
        elif row[2] == 'PrivateTest':
            pixels_test = [float(x) for x in row[1].split(' ')]
            imgs_test_byte.append(np.array(pixels_test))
            labels_test_class.append(int(row[0]))


# cast to correct type and wrap into np arrays
imgs_train_byte = np.array(imgs_train_byte, dtype=np.float32)
imgs_test_byte = np.array(imgs_test_byte, dtype=np.float32)
labels_train_class = np.array(labels_train_class, dtype=np.float32)
labels_test_class = np.array(labels_test_class, dtype=np.float32)
# normalize the pixel intensitys
imgs_train = np.divide(imgs_train_byte, 255)
imgs_test = np.divide(imgs_test_byte, 255)


# Build the Estimator
estimator = tf.estimator.Estimator(
    model_fn=model.model_fn, 
    model_dir='./model',
    params={
        'learning_rate': 0.001,
        'num_classes': 7,
        'dropout_rate': 0.25
    })

# Training
# Define the input function for training
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': imgs_train}, y=labels_train_class,
    batch_size=batch_size, num_epochs=None, shuffle=True)
# Train the Model
estimator.train(input_fn, steps=num_steps)



# Evaluate the Model
# Define the input function for evaluating
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': imgs_test}, y=labels_test_class,
    batch_size=batch_size, shuffle=False)
# Use the Estimator 'evaluate' method
estimator.evaluate(input_fn)