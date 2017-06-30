# -*- coding: utf-8 -*-

""" Convolutional network applied to CIFAR-10 dataset classification task.
References:
    Learning Multiple Layers of Features from Tiny Images, A. Krizhevsky, 2009.
Links:
    [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

    CSAL4243 Introduction to Machine Learning's assignment 3.
"""
from __future__ import division, print_function, absolute_import

import tflearn

from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from sklearn.model_selection import train_test_split

# load test images
from tflearn.data_utils import image_preloader
import numpy as np
# Load path/class_id image file:
dataset_file = 'dataset.txt'


image, label = image_preloader(dataset_file, image_shape=(100, 100), mode='file',
                       categorical_labels=True, normalize=True,
                       files_extension=['.jpg', '.png'], filter_channel=True)
image = np.array(image)
label = np.array(label)

X_train,X_test,y_train,y_test = train_test_split(image,label,test_size=0.2)

# Real-time data preprocessing
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Real-time data augmentation
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)

# Convolutional network building
## To Do.
    ## Define your network here
network = input_data(shape=[None, 100, 100, 3],
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug)
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 32, 3, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 32, 3, activation='relu')
network = max_pool_2d(network, 2)
network = fully_connected(network, 256, activation='relu')
network = dropout(network, 0.75)
network = fully_connected(network, 256, activation='relu')
network = dropout(network, 0.75)
network = fully_connected(network, 5, activation='softmax')
network = regression(network, optimizer='adam',loss='categorical_crossentropy',learning_rate=0.001)

# Train using classifier
## To Do
    ## Define model and assign network
    ## Call the fit function for training

model = tflearn.DNN(network, tensorboard_verbose=0)

model.fit(X_train, y_train, n_epoch=50, shuffle=True, validation_set=(X_test, y_test),
show_metric=True, batch_size=64, run_id='vehicleDetection')


# Manually save model
## To Do
    ## Save model

#model.save('project.tfl')
