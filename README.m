%%README.m


% This network is a fully connected feedforward neural network.

% The network's hyperparameters can be changed in the "assembleNetwork"
% function, which will create and trained a new network.

% The network uses the sigmoid activation function and a cross-entropy cost
% function, though backprop.m does contain a line with the alternate
% quadratic cost function.  
% The cross-entropy cost function has a noticeably worse performance with a
% higher learning rate, but will obtain consistently greater accuracy if
% that hyperparameter is set correctly.

% The network is intended for use on the contained MNIST dataset, but may
% be edited to handle other data.

% Assuming use on MNIST, the input and output layers must have 784 and 10
% neurons respectively.  The optimum network dimensions are then 1 or 2
% hidden layers with 30-40 neurons.  

% trainedNetOne - trainedNetFour are shallow FFNNs with 30 or 35 neurons in
% the single hidden layers.
% deepFFnetOne is a "deep" FFNN utilizing two hidden layers.

% The script "DemoTrainedNetwork" is a MATLAB app to visually demonstrate
% the network performing in real time on randomly shuffled test data.