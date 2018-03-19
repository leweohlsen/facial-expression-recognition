#!/usr/bin/env python

import numpy as np
import tensorflow as tf
import pandas as pd

import model

# Training Parameters.
max_steps = 1500
batch_size = 128
eval_interval = 240

# Read CSV with pandas.
ferplus = pd.read_csv('data/ferplus.csv')

# Split and cast pixel intensities to float32.
ferplus.pixels = ferplus.pixels.str.split()
ferplus.pixels = ferplus.pixels.map(lambda p: pd.to_numeric(p, downcast='float'))

# Filter noface class.
ferplus = ferplus.query('NF==0')

# Get argmax of class distribution to use as label.
labels = ferplus[['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']]
maxlabels = labels.idxmax(axis=1).map(labels.columns.get_loc)
ferplus.insert(loc=12, column='maxlabel', value=maxlabels)

# Split train, test and validation set.
train = ferplus.loc[ferplus['Usage'] == 'Training']
valid = ferplus.loc[ferplus['Usage'] == 'PublicTest']
test = ferplus.loc[ferplus['Usage'] == 'PrivateTest']

x_test = np.array(test['pixels'].values.tolist())
x_train = np.array(train['pixels'].values.tolist())
x_valid = np.array(valid['pixels'].values.tolist())

y_test = test['maxlabel'].values
y_train = train['maxlabel'].values
y_valid = valid['maxlabel'].values


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


# Define the input function for training.
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': x_train}, y=y_train,
    batch_size=batch_size, num_epochs=None, shuffle=True)

# Define the input function for validation during training.
valid_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': x_valid}, y=y_valid,
    batch_size=batch_size, num_epochs=None, shuffle=False)

# Specify training operations.
train_spec = tf.estimator.TrainSpec(
    input_fn = train_input_fn,
    max_steps = max_steps
)

# Specify evaluation operations.
eval_spec = tf.estimator.EvalSpec(
    input_fn = valid_input_fn,
    throttle_secs=eval_interval,
    start_delay_secs=eval_interval,
)

# Train the Model and evaluate periodically
tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


# Define the input function for evaluating with the test set.
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': x_test}, y=y_test,
    batch_size=batch_size, shuffle=False)
# Evaluate the model accuracy with the test set
estimator.evaluate(input_fn)

# Export the model as a SavedModel for production use
feature_spec = {'images': tf.placeholder(dtype=tf.float32, shape=[None, 48 * 48])}
serving_input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feature_spec)

estimator.export_savedmodel(
    export_dir_base='saved_models/ferplus',
    serving_input_receiver_fn=serving_input_fn,
    as_text=True)
