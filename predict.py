import tensorflow as tf
import numpy as np
import model

session = tf.Session()

# Build the Estimator
estimator = tf.estimator.Estimator(model_fn=model.model_fn, model_dir='./model')

def predictEmotion(img):
    # Evaluate the Model
    # Define the input function for evaluating
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'images': img}, shuffle=False)

    # Use the model to predict the images class
    preds = list(estimator.predict(input_fn))
    print(preds, " -> ", model.class_labels[preds[0]])
    return preds[0]