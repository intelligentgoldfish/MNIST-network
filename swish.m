%%swish.m
%%Swish activation function
%%Author: Thomas DeWitt

function z = swish(y)

z = y .* sigmoid(y);

end