# Object Recognition cifar-10 dataset
This repository contains the deep learning CNN model for object recognition in images from CIFAR-10 dataset. 

The implementation is done in python using Tensor flow.

# Libraries/ dependencies
1. PIL
2. numpy
3. tensorflow
4. math
5. pickle

# Description of the code files
1. process_data.py: In this file data is preprocessed. The image data is being converted from a pickle file into arrays.
source dataset CIFAR-10 can be downloaded from https://www.cs.toronto.edu/~kriz/cifar.html

2. cnn_netwrok.py: In this file the convolution neural network is being implemented from the data preprossed in process_data.py file

# Implementation Details:
The deep learning network consists of 4 layers. A 3 layer 2D convolution netwrok along with a fully connected last year is used.
The network prameters are initialized with Xavier initialization.
After some hyperparamter tuning the values obtained for few hypermeters are as follows: learning rate = 0.001, epochs=60, mini batch size= 64
An accuracy of around 70% is acheived with this simple convolution neural network

