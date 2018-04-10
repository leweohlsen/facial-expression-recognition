# Facial Expression Recognition

a convolutional neural network for recognizing facial expressions on live webcam images.

## Installation
The implementation has been tested with Python 3.6.3. You can create a fresh virtual environment with conda or virtualenv if you like. TensorFlow is [not supporting](https://www.tensorflow.org/install/install_windows#installing_with_anaconda) `conda` officially, so `pip` is used for package management. All the dependencies can be found in the `requirements.txt` file.

With your Python 3 environment activated, you can install the requirements with
```
pip install -r path/to/requirements.txt
```

## Live prediction
If your computer has a webcam, you can  compute predictions on the fly. Take off your glasses and hats and start the live-prediction with
```
python webcam.py
```

## Training
If you would like to train the tensorflow CNN yourself, you need to obtain the [FER2013 dataset](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) from kaggle and, optionally, the [CK+ dataset](http://consortium.ri.cmu.edu/ckagree/index.cgi). For CK+, you can use the `ckplus_to_csv.py` script to automatically detect all the faces, parse the grayscale intensities and collect all the CK-images into a single CSV file.

The directory structure should look like this:
```
project root
│   README.md
│   ...   
└───data
│   │   fer2013.csv
│   │   ckplus.csv
│   │
│   └───ckplus
│       │   images
│       │   labels
```
You can invoke the (somewhat lengthy) training process with
```
python train_fer.py
```

