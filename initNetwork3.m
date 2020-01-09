%%initNetwork.m
%%Initialize feedforward network
%%Version 3.2.2
%%Weight initialization optimized and smoothed for faster learning
%%Author: Thomas DeWitt

function [w,b] = initNetwork3(dim)

rng('shuffle');

numLayers = length(dim);

w = cell(1,numLayers-1); %preallocate storage for weights
b = cell(1,numLayers-1); %preallocate storage for biases

%initializes weights and biases using normal distribution
for n = 1:numLayers-1
    w{n} = randn(dim(n+1),dim(n))./sqrt(dim(n));   %initializes weights along neuron rows at one per neuron
    b{n} = randn(dim(n+1),1); %initializes biases along neural connections
    
    %w{n} = zeros(dim(n+1),dim(n)); %init to 0 for debug
    %b{n} = zeros(dim(n+1),1); %init to 0 for debug
end

end