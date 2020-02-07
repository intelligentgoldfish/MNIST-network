%%feedForward.m
%%Feedforward algorithm
%%Version 3.0
%%Author: Thomas DeWitt

function [a,z] = feedForward3(input,w,b,dim)

if size(input, 2) ~= 1 || size(input,1) ~= dim(1)
    error('Input is not correct size for this network');
end

numLayers = length(dim);

a = cell(1,numLayers); %capture outputs of each neuron
z = cell(1,numLayers); %capture pre-sigmoid outputs of each neuron

a{1} = input; %feed information into leading neurons
z{1} = input; %feed information into leading neurons

for n = 2:numLayers %begin at 2 as first layer only feeds information forward
    x = w{n-1} * a{n-1} + b{n-1}; % w{n-1}(1,1:dim(n-1))
    z{n} = x; %store pre-sigmoid outputs
    a{n} = swish(x); %store neuron outputs and feed to next layer
end

end