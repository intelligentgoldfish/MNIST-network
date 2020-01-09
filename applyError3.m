%%applyError3.m
%%Make changes in network according to error data and learning rate
%%Version 3.2.2
%%Error averaging moved out to updateBatch3
%%Author: Thomas DeWitt

function [newW,newB] = applyError3(w,b,dim,wError,bError,learningRate)

numLayers = length(dim);

newW = cell(numLayers - 1, 1);

for n=1:numLayers - 1
    newW{n} = zeros(dim(n+1),dim(n));
end

newB = cell(numLayers-1,1);

for n = 2:numLayers
    newB{n} = zeros(dim(n),1);
end

%adjust network
for n = 1:numLayers-1
    wT = -1 .* (learningRate .* wError{n});
    newW{n} = w{n} + wT;
    
    bT = -1 .* (learningRate .* bError{n});
    newB{n} = b{n} + bT;
end

end