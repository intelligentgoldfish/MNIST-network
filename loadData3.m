%%loadData3.m
%%Data loader for MNIST network
%%Version 3.1
%%Updated to cellularize data for cleaner formatting and better performance
%%Author: Thomas DeWitt

function [trainData,testData] = loadData3()

load vectorizedData train
load vectorizedData test

trainData = prepData(train,true);
testData = prepData(test,false);

end

function newData = prepData(data,vectorize)

dataSize = size(data,1);

newData = cell(dataSize,2); %each row contains data cell and label cell

labels = data(:,1);
images = data(:,2:end);

labels(labels == 0) = 10; %change '0' labels to '10' for indexing
images = images./256; %scale image values for network

for n = 1:dataSize
    newData{n,1} = transpose(images(n,:));
%     getData = data(n,2:end)'; %note transpose
%     newData{n,1} = getData;
end

if vectorize == true
    %vectorize label for backprop algorithm
    for n = 1:dataSize
        thisLabel = zeros(10,1);
        thisLabel(labels(n,1)) = 1;
        newData{n,2} = thisLabel;
%         thisLabel = labels(n,1);
%         labelVector = zeros(10,1);
%         labelVector(thisLabel+1,1) = 1;
%         newData{n,2} = labelVector;
    end
else
    %keep label
    for n = 1:dataSize
        newData{n,2} = labels(n,1);
    end
end

end