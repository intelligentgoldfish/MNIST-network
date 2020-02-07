%%logit.m
%%Inverse sigmoid (logit) function

%%Author: Thomas DeWitt

function y = logit(z)

y = log(z ./ (1-z));

end