# Vehicle-Detection-with-Convolution-Neural-Network-

# Overview

The objective of this project is to train CNN model on images of different vehicles and using sliding window approach to detect different types of vehilces in the image.

# Dependencies

Download and Install the following packages
- tensorflow
- tflearn
- h5py
- hdf5
- SciPy
- numpy
- cv2

# Dataset

 We capture the video near the local traffic signal using camera and use matlab to extract images from the video. 

## Traing Data
After extracting the images from video using crop_data.m file we save the images in a separate folder and labeled them as
- 0 for rikshaws. 
- 1 for cars.
- 2 for bikes
- 4 for trucks/buses.
- 3 for non vehicles.

In our case datatset contains 951 images in which 127 images were labeled as 0, 409 images were labeled as 1, 195 images were labeled as 2 and 219 images were labeled as 3 and the rest ones were labeled as 4.

check the file format <b>dataset.txt</b>

Some positive and negative images are as follows:

### Positive Image
![positive](https://user-images.githubusercontent.com/26170668/27723536-139483ac-5d87-11e7-9777-4f860b7a90d5.png).

### Negative Image
![nagative](https://user-images.githubusercontent.com/26170668/27723538-19226ce4-5d87-11e7-8922-cb055850617f.png).

## Test Data

Images containing multiple vehicles extracted from another video not used for training.

# How to Run the Model

Follwing are the files to run the model
1. train_project.py: Used to train the data.
2. test.py:Load the image for testing and detects a vehicles using sliding window approach.
3. dataset.txt: set of images in a formatted order .
4. crop_data.m: matlab code for cropping imaes from video.


  First run the matlab code on the captured video to detect images and create a dataset file named 
  as dataset.txt and then run the files by typing the follwoing commands on terminal
  
  python train_project.py  
  python test.py      


# Implementation

## Convolution Neural Network (CNN)
We use CNN in our case as CNN is best fit where we have dataset of images.
### Layers
1. Input data shape= [100*100,3]
2. Conv: 64 filters of size 3x3 with ReLU activation
3. Pooling: with filter size 2x2
4. Conv: 32 filters of size 3x3 with ReLU activation
5. Pooling: with filter size 2x2
6. Conv: 32 filters of size 3x3 with ReLU activation
7. Pooling: with filter size 2x2
8. Fully Connected: with 256 neurons and ReLU activation and dropout with probability 0.75
9. Fully Connected: with 256 neurons and ReLU activation and dropout with probability 0.75
10. Fully Connected output layer: with 5 neurons (equal to number of classes) and softmax classifier.


## Load Dataset
 tflearn image preloader was used to load train dataset using a file 
 

# Result

## Training
    
We ran the cnn model for 5 epox and got accuracy 0.8398 with validation accuracy of 0.8796 using learning date of 
0.001 with adam optimizer and 80% data used for tsing and 20% data used for validation.
   
   ![result1](https://user-images.githubusercontent.com/26170668/27747443-93bca692-5de4-11e7-9fdf-c4300dda9f22.png)

   



Run the cnn model for 20 epox which gives accuracy 0.9454 with validation accuracy of 0.8586 using learning date of 0.001 with adam
optimizer
   
   ![result2](https://user-images.githubusercontent.com/26170668/27747425-7efdab02-5de4-11e7-8d2e-b030ca6473c7.png)



## Testing on Image

This is our originl Image on which we perform testing.

![test1](https://user-images.githubusercontent.com/26170668/27746961-bd8c3886-5de2-11e7-98f5-7a4842fad7b5.jpg)

   sliding window apporach is used having width and height 100*100. 
   
   
 ![all_rec](https://user-images.githubusercontent.com/26170668/27745828-4b7d795c-5dde-11e7-977d-a4f8df512b95.png)


Than we perform the same function on our original Image and than perform testing and we got the following result.
![test](https://user-images.githubusercontent.com/26170668/27746960-bd4fad58-5de2-11e7-8b11-fbcbb15cbf20.jpg)


After using all_rec we got many rectangles. To avoid this we use group_rec function here and got following results.
![result](https://user-images.githubusercontent.com/26170668/27745882-7e84bc8e-5dde-11e7-876f-2d21c55d389c.png)

# Discussion

This project aims to detect vehicles on unseen data using cnn network. The difficulty faced so far was to capture the video of atleast 25 minutes so that atleast 1000 images can be cropped down from the video. cropping is done using matlab code but it requires selection point for each frame so it was a difficult task to do. Moreover cnn network using 5 and 20 epox gives different accuracies which was a challenge as well.However to produce better result(more accuracy) we can increase the epox and try different learning rates.


# Reference

https://w4zir.github.io/ml17s/


