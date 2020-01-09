%%assembleNetwork.m
%%Command-line code for creating and testing neural network
%%Version 3.2.2
%%Author: Thomas DeWitt

clear; clc;

disp('~~~MNIST-Trainable Feedforward Neural Network~~~');
disp('Author: Thomas DeWitt');
disp('Version 3.2.2');
disp(['Updates:',newline,'-Tweaked and smoothed initialization for faster learning',newline,...
    '-Removed extraneous code and commenting',newline,...
    '-Streamlined and condensed existing code',newline,newline,...
    'Press any key to create and train a new network.',newline]);

pause;

disp('Initializing...');

dim = [784 35 10];
epochs = 20;
miniBatchSize = 10;
learningRate = 3;

[w,b] = initNetwork3(dim);

disp(['Done.',newline,'Training...']);

[finalWeights,finalBiases,accuracy] = SGD3(w,b,dim,epochs,...
    miniBatchSize,learningRate);

disp('Training complete.');
disp(['Average accuracy: ',num2str(accuracy),'%']);

% disp(['Training complete.',newline,'Conducting final test...']);
% 
% [~,~,testData,testLabels] = loadData();
% [stat,~] = testNetwork(w,b,dim,testData,testLabels);
% 
% disp(['Done.',newline,'Accuracy: ',num2str(stat),'%']);