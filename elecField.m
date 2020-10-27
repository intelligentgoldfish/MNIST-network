%%elecField.m
%%Electric field simulator
%%V1.0.0

freq = 0.5;
maxes = 5;
mins = -5;

x = mins:freq:maxes;
y = x;

[x,y] = meshgrid(x,y);

k = 9*(10^9);
q = 1e-12;
p = [1 1];
x1 = x-p(1);
y1 = y-p(2);

Ex = (k.*q.*x1)./(x1.^2+y1.^2).^(3/2);
Ey = (k.*q.*y1)./(x1.^2+y1.^2).^(3/2);

figure;
quiver(x,y,Ex,Ey);