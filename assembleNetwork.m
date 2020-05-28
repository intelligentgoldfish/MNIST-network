%%DenseMatNet
%%Version 4.1.2
%%Feel free to modify any of the contents of this file.
%%Please cite the author of these programs if presenting or using them
%%elsewhere

%%Author: Thomas DeWitt
%%Last modification: 5/27/2020 11:05pm

function assembleNetwork()

clear; clc;

disp('~~~DenseMatNet (Physics Variant)~~~');
disp('Author: Thomas DeWitt');
disp('Version 4.1.2');
disp(['Updates:',newline,'-Changed margin of error to 5cm',newline,...
    '-Reduced learning rate and tripled epochs',newline,newline,...
    'Press any key to create and train a new network.',newline]);

pause;

disp('Initializing...');

dim = [2 35 1];
epochs = 70;
miniBatchSize = 10;
learningRate = 0.1;

[w,b] = initNetwork3(dim);

disp(['Done.',newline,'Training...']);

%weights and biases saved internally by SGD3 function
[~,~,accuracy] = SGD3(w,b,dim,epochs,miniBatchSize,learningRate);

disp('Training complete.');
disp(['Average accuracy: ',num2str(accuracy),'%',newline]);

disp([newline,'Remember to edit the name of the saved network to avoid',...
    newline,'overwriting it next time!']);

end



%% Stochastic Gradient Descent function

function [newW,newB,avgAccuracy] = SGD3(w,b,dim,epochs,miniBatchSize,learningRate)

disp('Prepping data...');
%[train,test] = loadData3();
load pendata trainData testData
train = trainData;
test = testData;

rng('shuffle');
disp(['Done.',newline,'Commencing epochs...',newline]);

numBatches = size(train,1)/miniBatchSize;

avgAccuracy = 0;

disp('Starting timer...');
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
        [eW,eB,cost] = updateBatch3(w,b,dim,thisBatch);
        [w,b] = applyError3(w,b,dim,eW,eB,learningRate);
    end
    
    correct = 0;
    for k = 1:size(test,1)
        output = getOutput(test{k,1},w,b,dim);
        %margin of error: +-0.05m (5cm)
        if output > test{k,2} - 0.05
            if output < test{k,2} + 0.05
                correct = correct + 1;
            end
        end
    end
    stat = 100*correct/size(test,1);
    
    avgAccuracy = avgAccuracy + (correct/size(test,1));
    
    timeElapsed = toc;

    %display progress
    disp(['Epoch ',num2str(epoch),' of ',num2str(epochs),' complete.']);
    disp(['Time elapsed: ',num2str(timeElapsed),' seconds.']);
    disp([num2str(stat),'% accuracy.']);
    disp(['Average cost: ',num2str(cost),newline]);
end

avgAccuracy = 100 * avgAccuracy / epochs;

newW = w;
newB = b;

save lastTrainedNetwork w b dim avgAccuracy

end


%% Mini-Batch Processing function

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


%% Backpropagation Algorithm

function [dW,dB,cost] = backprop3(a,z,w,dim,label)

%ensure data has been correctly formatted
if size(label, 2) ~= 1 || size(label,1) ~= dim(end)
    error('Desired output is not correct size for this network.  Check data formatting.');
end

numLayers = length(dim);

gradient = (a{numLayers} - label); %gradient

cost = sum(gradient.^2)/2; %quadratic cost function

%desired change in output
%delta = gradient .* sigmoidPrime(z{numLayers}); %quadratic cost
delta = gradient; %cross-entropy cost

for n = (numLayers-1):-1:1 %for each layer backpropagating
    dW{n} = delta * a{n}'; %calc and store weight error
    dB{n} = delta; %store bias error
    delta = w{n}' * delta .* sigmoidPrime(z{n}); %calc error in next layer
end

end


%% Error Application function

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


%% Forward Pass function

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


%% Random network initialization

function [w,b] = initNetwork3(dim)

rng('shuffle');

numLayers = length(dim);

w = cell(1,numLayers-1); %preallocate storage for weights
b = cell(1,numLayers-1); %preallocate storage for biases

%initializes weights and biases using normal distribution
for n = 1:numLayers-1
    w{n} = randn(dim(n+1),dim(n))./sqrt(dim(n));   %initializes weights along neuron rows at one per neuron
    b{n} = randn(dim(n+1),1); %initializes biases along neural connections
end

end


%% Network output interpreter

function output = getOutput(input,w,b,dim)

numLayers = length(dim);

[a,~] = feedForward3(input,w,b,dim);

output = a{numLayers};

end


%% Sigmoid function

function y = sigmoid(z)

    y = 1./(1+exp(-1.*z));

end


%% Sigmoid derivative

function y = sigmoidPrime(z)

y = sigmoid(z).*(1-sigmoid(z));

end


%% Network mapping function

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

%display image
map1 = figure(1);
figure(map1);
%sometimes MATLAB will generate complex numbers which it shouldn't
firstLayerMap = reshape(real(a{1}),28,28)';
imagesc(firstLayerMap);
colormap jet;

end


%% Inverse sigmoid for network mapping

function y = logit(z)

y = log(z ./ (1-z));

end