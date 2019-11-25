%%feedForward.m
%%Takes an input x, the weights, and the biases of the entire network
%%Feeds information through network
%%Returns all activations and pre-sigmoid activations

%%Author: Thomas DeWitt

function [activations,layerOutputValues] = feedForward(x,weights,biases,dim)

if size(x, 2) ~= 1 || size(x,1) ~= dim(1)
    error('Input is not correct size for this network');
end

numLayers = length(dim);

activations = cell(1,numLayers); %capture outputs of each neuron
layerOutputValues = cell(1,numLayers); %capture pre-sigmoid outputs of each neuron

activations{1} = x; %feed information into leading neurons
layerOutputValues{1} = x; %store information for analysis

for n = 2:numLayers %begin at 2 as first layer only feeds information forward
    z = dot(weights{n-1}(1,1:dim(n-1)),activations{n-1}) + biases{n-1}; % * gives dot product here as weight layers are vectors
    layerOutputValues{n} = z; %store pre-sigmoid outputs
    activations{n} = sigmoid(z); %store neuron outputs and feed to next layer
end

end