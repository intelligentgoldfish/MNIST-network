%%SGD3.m
%%Stochastic gradient descent algorithm for feedforward neural network
%%Version 3.3.1
%%Utilizing loadData3 to cellularize data
%%Author: Thomas DeWitt

function [newW,newB,avgAccuracy] = SGD3(w,b,dim,epochs,miniBatchSize,learningRate)

disp('Prepping data...');
[train,test] = loadData3();

rng('shuffle');
disp(['Done.',newline,'Commencing epochs...',newline]);

numBatches = size(train,1)/miniBatchSize;

avgAccuracy = 0;

tic; %initialize chronometer

for epoch = 1:epochs
    
    %shuffle data and maintain label order
    shuffleOrder = randperm(size(train,1));
    train = train(shuffleOrder,:);

    for m = 1:numBatches
        %pull data sequentially for each mini-batch
        batchStart = 1 + (m-1) * miniBatchSize; 
        batchRange = batchStart:(batchStart+miniBatchSize-1);
        thisBatch = train(batchRange,:);
        
        %backprop and tweak
        [eW,eB,~] = updateBatch3(w,b,dim,thisBatch);
        [w,b] = applyError3(w,b,dim,eW,eB,learningRate);
    end
    
    correct = 0;
    for k = 1:size(test,1)
        output = identifydigit3(test{k,1},w,b,dim);
        if output == test{k,2}
            correct = correct + 1;
        end
    end
    stat = 100*correct/size(test,1);
    
    avgAccuracy = avgAccuracy + (correct/size(test,1));
    
    timeElapsed = toc;

    meanTime = timeElapsed/epoch;
    estimatedTime = meanTime*(epochs-epoch);

    %display progress
    disp(['Epoch ',num2str(epoch),' of ',num2str(epochs),' complete.']);
    disp(['Time elapsed: ',num2str(timeElapsed),' seconds.']);
    disp([num2str(stat),'% accuracy.']);
    disp(['Estimated time remaining: ',num2str(estimatedTime),newline]);
end

avgAccuracy = 100 * avgAccuracy / epochs;

newW = w;
newB = b;

save lastTrainedNetwork w b dim avgAccuracy

end