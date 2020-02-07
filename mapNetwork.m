%%feedBackward.m
%%Maps the critical locations of a fully-connected neural network

%%Author: Thomas DeWitt

function [a,z] = mapNetwork(w,b,dim,shuffle)

numLayers = length(dim);

if shuffle
    rng('shuffle');
end

%create random noise
digit = rand(dim(numLayers),1);

a = cell(1,numLayers);
z = cell(1,numLayers);

a{numLayers} = digit;
z{numLayers} = digit;

for n = numLayers-1:-1:1 %feeding backwards
    x = logit(a{n+1});
    z{n} = x;
    a{n} = transpose(w{n}) / (x - b{n})'; %align matrices correctly
end

%sometimes MATLAB will generate complex numbers which it shouldn't

%display image
map1 = figure(1);
figure(map1);
firstLayerMap = reshape(real(a{1}),28,28)';
imagesc(firstLayerMap);
colormap jet;

end