# MNIST-network
MATLAB code for neural network to analyze handwritten numbers, trainable off of the MNIST dataset.


# About
This is a simple, adjustable feedforward neural network implemented in MATLAB and trainable off of the MNIST dataset.

The network was created over a period of three weeks in addition to other classes during 11th grade.  It has not been written for maximum possible performance, but is instead written to be easy to read, understand, and alter.

Feel free to alter portions of the code, especially activation and cost functions.

# Code Interactions

The structure of the code is as follows:

assembleNetwork.m will create a completely new network, with quickly adjustable parameters such as network dimensions, mini-batch size, number of epochs, and learning rate.  After loading the training and testing data, it will initialize random weights and biases to fit the desired network dimensions, smoothing the weights and biases slightly to improve the initial training speed.
Immediately after initializing, assembleNetwork.m passes control to the function SGD3.m, which executes the stochastic gradient descent algorithm.  For each epoch, SGD3.m will shuffle the training data, divide it into mini-batches, and then pass each mini-batch to updateBatch3.m for training.
For each new piece of data, updateBatch3.m will run the data through the network (feedForward3.m) and backpropagate (backprop3.m).  It saves the error and cost and averages them across the entire mini-batch before returning control to SGD3.m, which then gives the data to applyError3.m to use.  applyError3.m takes the error data and adjusts the weights and biases of the network accordingly.
At the end of each epoch, the network will test itself and return the percentage correct, the average cost, and the elapsed time since the network began training.
The network will automatically save itself in a file called lastTrainedNetwork.mat.  I suggest changing the name of this file if you obtain a network you wish to keep, or you will be constantly overwriting your old networks.
Finally, the network will return its average accuracy.  This may be slightly lower than the actual accuracy but will come very close.  This is a potential flaw which I intend to rectify, but have not yet disposed of.

# Critical Scripts

The following scripts contain core functions and hyperparameters of the network:

backprop3.m (contains cost function and derivative of activation function)
feedForward3.m (contains activation function)
assembleNetwork.m (contains dimensions, learning rate, epochs, mini-batch size)
initNetwork3.m (creates new network with random weights and biases)
sigmoid.m/sigmoidPrime.m (sigmoid function/derivative)
