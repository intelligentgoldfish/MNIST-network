%%swishPrime.m
%%Derivative of swish function
%%Author: Thomas DeWitt

function z = swishPrime(y)

z = swish(y) + sigmoid(y) .* swish(1-y);

end