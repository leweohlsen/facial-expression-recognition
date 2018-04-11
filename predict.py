import tensorflow as tf
from tensorflow.contrib import predictor
import numpy as np
import model

class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# fer2013_predictor = predictor.from_saved_model(export_dir='saved_models/fer2013/1520859040')
ferplus_predictor = predictor.from_saved_model(export_dir='saved_models/ferplus/1523392049')
# ckplus_predictor = predictor.from_saved_model(export_dir='saved_models/ckplus/1520882345')

def predictEmotion(img_128, img_48):
    # output_dict_fer2013 = fer2013_predictor({'images': img_48})
    output_dict_ferplus = ferplus_predictor({'images': img_48})
    # output_dict_ckplus = ckplus_predictor({'images': img_128})
    # print("CK+ = ", output_dict_ckplus['output'][0])
    # print("FER = ", output_dict_fer2013['output'][0])
    print("FER+ = ", output_dict_ferplus['output'][0])
    print(output_dict_ferplus)
    return output_dict_ferplus['output'][0]

def predict_emotion_probas(img_128, img_48, n=3):
    output_dict_ferplus = ferplus_predictor({'images': img_48})
    class_indices = np.argsort(output_dict_ferplus['output'][0])
    return (class_indices[::-1], output_dict_ferplus['output'][0])