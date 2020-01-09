%%identifyDigit3
%%MNIST digit identification
%%Version 3.0
%%Internalizes feedforward algorithm instead of requiring external feeding
%%Author: Thomas DeWitt

function digit = identifydigit3(input,w,b,dim)

numLayers = length(dim);

[a,~] = feedForward3(input,w,b,dim);

output = a{numLayers};

[~,answer] = max(output);

digit = answer;

end