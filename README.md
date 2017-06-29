# Vehicle-Detection-with-Convolution-Neural-Network-

# Deep Learning

# Overview

The objective of this project is to dectect vehicle and non vehicle from image using sliding window approach. Train CNN model on dataset to acheive this objective.

# Dependencies

- tensorflow
- tflearn
- h5py
- hdf5
- SciPy
- numpy
- cv2

# How to Run the Model

This repository comes with trained model which you can directly test using the following command.

python test.py

## Implementation

### Model
# Layers
1. Conv: 64 filters of size 3x3 with ReLU activation
2. Pooling: with filter size 2x2
3. Conv: 32 filters of size 3x3 with ReLU activation
4. Pooling: with filter size 2x2
5. Conv: 32 filters of size 3x3 with ReLU activation
6. Pooling: with filter size 2x2
7. Fully Connected: with 256 neurons and ReLU activation and dropout with probability 0.75
8. Fully Connected: with 256 neurons and ReLU activation and dropout with probability 0.75
9. Fully Connected output layer: with 2 neurons (equal to number of classes) and softmax classifier.

# Train Dataset

create dataset from video, capture video from any enviromnet where Vehicle and non vehicle then crop vehicle and non vehile from every frame check the file for crop image <b>crop_data.m<b>

when extrat data from video using crop_data.m file it Save picture in Data folder. create a new file and save record as
0 for rikshaws.
1 for cars.
2 for bikes
4 for trucks/buses.
3 for non vehicles.

check the file format <b>dataset.txt<b>

# Test Dataset

Take picture fom any enviroment or select image from dataset fro testing


