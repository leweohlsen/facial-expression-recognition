# Facial Expression Recognition

a convolutional neural network for recognizing facial expressions on live webcam images.

## Training
If you would like to train the tensorflow CNN yourself, you need to obtain the [FER2013 dataset](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) from kaggle and, optionally, the [CK+ dataset](http://consortium.ri.cmu.edu/ckagree/index.cgi). For CK+, you can use the `ckplus_to_csv.py` script (util-directory) to automatically detect all the faces, parse the grayscale intensities and collect all the CK-images into a single CSV file.

With the our directory structure should look like this:
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
- `python train.py`

## Live prediction
- `python webcam.py`

