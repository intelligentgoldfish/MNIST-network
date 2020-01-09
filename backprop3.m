%%backprop3.m
%%Backpropagation algorithm for feedforward neural network
%%Version 3.2.2
%%Removed unnecessary input arguments and trimmed code
%%Author: Thomas DeWitt

function [dW,dB,cost] = backprop3(a,z,w,dim,label)

%ensure data has been correctly formatted
if size(label, 2) ~= 1 || size(label,1) ~= dim(end)
    error('Desired output is not correct size for this network.  Check data formatting.');
end

numLayers = length(dim);

gradient = (a{numLayers} - label); %gradient

cost = sum(gradient.^2)/2; %quadratic cost function

%desired change in output
delta = gradient .* sigmoidPrime(z{numLayers});

for n = (numLayers-1):-1:1 %for each layer backpropagating
    dW{n} = delta * a{n}'; %calc and store weight error
    dB{n} = delta; %store bias error
    delta = w{n}' * delta .* sigmoidPrime(z{n}); %calc error in next layer
end

end