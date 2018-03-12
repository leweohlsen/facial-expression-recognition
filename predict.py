import tensorflow as tf
from tensorflow.contrib import predictor
import numpy as np
import model

class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

fer2013_predictor = predictor.from_saved_model(export_dir='saved_models/fer2013/1520859040')
# ckplus_predictor = predictor.from_saved_model(export_dir='saved_models/ckplus/1520882345')

def predictEmotion(img):
    output_dict_fer2013 = fer2013_predictor({'images': img})
    # output_dict_ckplus = ckplus_predictor({'images': img})
    return output_dict_fer2013['output'][0]
