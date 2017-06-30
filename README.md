# Vehicle-Detection-with-Convolution-Neural-Network-

# Deep Learning

# Overview

The objective of this project is to dectect different types of vehicles from images using sliding window approach and then train CNN model on that dataset to detect vehicles from unseen data(generalization).

# Dependencies

- tensorflow
- tflearn
- h5py
- hdf5
- SciPy
- numpy
- cv2

# Dataset

capture video from any enviromnet where Vehicles and non vehicles are present.capture images from the video frame by frame using matlab code from the mentioned matlab file <b>crop_data.m</b>

when extrat data from video using crop_data.m file it Save the pictures in Data folder. create a new file and save record as
0 for rikshaws.
1 for cars.
2 for bikes
4 for trucks/buses.
3 for non vehicles.

check the file format <b>dataset.txt</b>

# How to Run the Model

## Training

run the model using cnn for 5 iterations 
run the model using cnn for 20 iterations 

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




# Test Dataset

Take picture fom any enviroment and load that image in <test.py> file 
and run the following command on the terminal

python test.py


# Result

run the model using cnn for 5 iterations which gives the following result.




run the model using cnn for 20 iterations which gives the following result.
.....

# Discussion
This project took 4 days to completePorject was divided into different tasks like capturing video, detection of vehicles (creating dataset) and train the cnn model on dataset for unseen data and testing on new image. All the tasks were completed on different days.

# Reference

https://w4zir.github.io/ml17s/


