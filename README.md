# Autoencoder-Model

Autoencoders are a type of artificial neural network that is used for unsupervised learning. It has two major components, encoder and decoder which determines the function of the autoencoder neural network. The goal of an autoencoder is to form a representation for a set of data, by ignoring the signal noise in the network by continously training on the data. An autoencoder has two major components:

I. Encoder - An encoder takes the input and compresses the data into a tensor called latent space.
II. Decoder - The decoder takes the latent space tensor representation and reconstructs the data back into its original form. The major goal is to make the output or the target data as close as possible to the input data.

The major applications of Autoencoders are:
1. Dimensionality Reduction (mostly used for decreasing the number of dimensions to make the learning process easier)
2. Noise removal from data.
3. Anomaly detection in text or dataset.

This repository contains the application of autoencoder in MNIST Handwritten Digit dataset to recreate the existing handwritten data with the highest accuracy as possible. This is however my first practical experience with autoencoders for which I prefered to work with the basic widely used handwritten digit dataset.

In order to test the autoencoder model, kindly ensure that you have python 3.7 and Tensorflow 2.0.0 and above working in your system.
train the algorithm using the command "python train.py". Your output shall be generated in the "recreated_data.png" file.
