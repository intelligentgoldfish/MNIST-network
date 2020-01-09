%%testNetwork3.m
%%Network test across 10,000 MNIST images
%%Author: Thomas DeWitt

function stat = testNetwork3(testData,w,b,dim)

correct = 0;
for k = 1:size(testData,1)
    output = identifydigit3(testData{k,1},w,b,dim);
    if output == testData{k,2}
        correct = correct + 1;
    end
end
stat = 100*correct/size(testData,1);
    
end