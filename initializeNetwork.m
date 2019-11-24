%%initializeNetwork.m
%%Initialization code for my neural network
%%Takes dimensions of network as a vector, with a number of elements equal to the number of layers, 
%%and each element being the number of neurons in the layer to which it is assigned

function [weights,biases] = initializeNetwork(dim)

    numLayers = length(dim);

    weights = cell(1,numLayers-1); %preallocate storage for weights
    biases = cell(1,numLayers-1); %preallocate storage for biases

    %initializes weights and biases using normal distribution
    for n = 1:numLayers-1
        weights{n} = randn(dim(n+1),1); %initializes weights along neuron rows at one per neuron
        biases{n} = randn(dim(n+1),dim(n)); %initializes biases along neural connections
    end

end
