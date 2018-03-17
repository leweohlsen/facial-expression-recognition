#!/usr/bin/env python

import numpy as np
import tensorflow as tf
import pandas as pd

import model

# Training Parameters
num_steps = 10000
batch_size = 128

# read csv with pandas
ferplus = pd.read_csv('data/ferplus.csv')

# split and cast pixel intensities to float32
ferplus.pixels = ferplus.pixels.str.split()
ferplus.pixels = ferplus.pixels.map(lambda p: pd.to_numeric(p, downcast='float'))

# filter noface class
ferplus = ferplus.query('NF==0')

# get argmax of class distribution to use as label
labels = ferplus[['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']]
maxlabels = labels.idxmax(axis=1).map(labels.columns.get_loc)
ferplus.insert(loc=12, column='maxlabel', value=maxlabels)

# split train, test and validation set
train = ferplus.loc[ferplus['Usage'] == 'Training']
test = ferplus.loc[ferplus['Usage'] == 'PublicTest']
validation = ferplus.loc[ferplus['Usage'] == 'PrivateTest']

x_train = np.array(train['pixels'].values.tolist())
x_test = np.array(test['pixels'].values.tolist())
x_validation = np.array(validation['pixels'].values.tolist())

y_train = train['maxlabel'].values
y_test = test['maxlabel'].values
y_validation = validation['maxlabel'].values


# Build the Estimator
estimator = tf.estimator.Estimator(
    model_fn=model.model_fn, 
    model_dir='./model/ferplus',
    params={
        'learning_rate': 0.001,
        'num_classes': 7,
        'img_size': 48,
        'dropout_rate': 0.25
})



# Training
# Define the input function for training
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': x_train}, y=y_train,
    batch_size=batch_size, num_epochs=None, shuffle=True)
# Train the Model
estimator.train(input_fn, steps=num_steps)



# Evaluate the Model
# Define the input function for evaluating
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': x_test}, y=y_test,
    batch_size=batch_size, shuffle=False)
# Use the Estimator 'evaluate' method
estimator.evaluate(input_fn)

# Export the model as a SavedModel for production use
feature_spec = {'images': tf.placeholder(dtype=tf.float32, shape=[None, 48 * 48])}
serving_input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feature_spec)

estimator.export_savedmodel(
    export_dir_base='saved_models/ferplus',
    serving_input_receiver_fn=serving_input_fn,
    as_text=True)
