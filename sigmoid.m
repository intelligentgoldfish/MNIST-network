%%sigmoid neuron algorithm
%%calculates the sigmoid function of z
%%Author: Thomas DeWitt

function y = sigmoid(z)

    y = 1/(1+exp(-1*z));

end