%%updateBatch3.m
%%Train and update network across a single mini-batch
%%Version 3.2
%%applyError3 function moved outside of program to SGD3 body
%%Error averaging internalized in function
%%Author: Thomas DeWitt

function [eW,eB,avgCost] = updateBatch3(w,b,dim,data)

numLayers = length(dim);
totalCost = 0;

ewSum = cell(1,numLayers-1);
ebSum = cell(1,numLayers-1);

%initialize errors to 0
for n = 1:numLayers-1
    ewSum{n} = zeros(dim(n+1),dim(n));
    ebSum{n} = zeros(dim(n+1),1);
end

eW = cell(1,numLayers-1);
eB = cell(1,numLayers-1);

batchSize = size(data,1);

for n = 1:batchSize
    input = data{n,1};
    output = data{n,2};
    
    %feedforward
    [a,z] = feedForward3(input,w,b,dim);
    
    %backpropagate
    [dW,dB,cost] = backprop3(a,z,w,dim,output);
    
    %add to sigma(dc/dw) and sigma(dc/db)
    for k = 1:numLayers-1
        ewSum{k} = ewSum{k} + dW{k};
        ebSum{k} = ebSum{k} + dB{k};
    end
    
    totalCost = totalCost + cost;
    
end

for n = 1:numLayers-1
    eW{n} = ewSum{n}./batchSize;
    eB{n} = ebSum{n}./batchSize;
end

avgCost = totalCost/batchSize;

end