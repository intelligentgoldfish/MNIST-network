%%sigmoidPrime.m
%%Derivative of sigmoid function
%%Author: Thomas DeWitt

function y = sigmoidPrime(z)

y = sigmoid(z).*(1-sigmoid(z));

end