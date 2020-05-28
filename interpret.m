%%interpret

function height = interpret(distance)

load network94 w b dim

plength = 1.97;
landed = 1;

%blength = 0.21;
%ramp_rem = 0.67;
%rlength = ramp_rem + blength/2;
%buckwidth = 0.237;
%buckdist = 1.675;
%fulldist = buckdist + rlength + buckwidth/2 - blength/2;

%k = fulldist/3.149261757715348;

%input = [landed k]';


distance = distance/3.149261757715348;
input = [landed distance]'; %note transpose

[a,~] = feedForward3(input,w,b,dim);
height = a{3} * plength;

disp(['Network-computed drop height: ',num2str(height),' meters']);

end

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
    a{n} = sigmoid(x); %store neuron outputs and feed to next layer
end

end

function y = sigmoid(z)

    y = 1./(1+exp(-1.*z));

end