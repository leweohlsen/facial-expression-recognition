import tensorflow as tf
from tensorflow.contrib import predictor
import numpy as np
import model

class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

saved_model_predictor = predictor.from_saved_model(export_dir='saved_models/1520693695')

def predictEmotion(img):
    output_dict = saved_model_predictor({'images': img})
    return output_dict['output'][0]