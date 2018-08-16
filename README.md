# Overview

This repo contains a port of the [TensorFlow Deep MNIST tutorial](https://github.com/tensorflow/tensorflow/blob/r1.9/tensorflow/examples/tutorials/mnist/mnist_deep.py). The tutorial code uses TensorFlow's Python API. It is the aim of this repo to port that to the C++ API so that I can compare runtimes. **Note: The code is not yet complete**.

# Current Status

The ported code is still a work in progress and there is quite a bit of test code in it. The following items have been addressed so far:

* Read in image and label data from MNIST files
* Build Tensors with the data
* Fetch a batch for the data
* Build a graph (although not all steps have been implemented yet)
* Run the graph

# Compilation

Currently, this project should be checked out and added as a sub-module in a tensorflow build. That is, this repo should be added as a repo to the root directory of a clone of the [TensorFlow repo](https://github.com/tensorflow/tensorflow).

It can then be built - along with TensorFlow - using [bazel](https://bazel.build/).

I plan to separate the build of this project from the TensorFlow build once the code is complete.

# Data

To run this project you will need to get a copy of the MNIST data set and add it to the 'data' sub-directory of the directory that contains this code.

