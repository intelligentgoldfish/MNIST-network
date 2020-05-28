%%Pendulum Generator
%%Thomas DeWitt

farthest = 3.149261757715348;


trainSize = 25000;
testSize = 4000;

trainData = cell(trainSize,2);
testData = cell(testSize,2);

plength = 1.97;

rng('shuffle');
for n = 1:trainSize
    x = rand;
    dropheight = plength * x;
    [landed,distance] = scenario(dropheight);
    distance = distance/farthest;
    k = [landed distance]';
    trainData{n,1} = k;
    trainData{n,2} = x;
end

rng('shuffle');
for n = 1:testSize
    x = rand;
    dropheight = plength * x;
    [landed,distance] = scenario(dropheight);
    distance = distance/farthest;
    k = [landed distance]';
    testData{n,1} = k;
    testData{n,2} = x;
end

save pendata trainData testData

function [landed,distance] = scenario(dropheight)

%gravity
g = 9.8;

%pendulum
pmass = 0.7704;

%bottle
bmass = 0.1494;
blength = 0.21;

%ramp
ramp_rem = 0.67;
%measure from middle of bottle
rlength = ramp_rem + blength/2;
rangle = 10; %ramp angle in degrees for static friction

%height metrics
rheight = 0.915;
buckheight = 0.372;
buckwidth = 0.237;
trueheight = rheight - buckheight;
buckdist = 1.675;
%fulldist = buckdist + rlength + buckwidth/2 - blength/2;
mindist = buckdist + rlength - blength/2;
maxdist = buckdist + rlength + buckwidth - blength/2;

fallTime = (2 * g * trueheight)^(1/2)/(g);

mu = tand(rangle);
Fn = bmass * g;
Ff = mu * Fn;

%air resistance
muA = 1;

h = dropheight;

simpleVp = (2 * g * h)^(1/2);

Vp = simpleVp * muA;

ma = bmass;
vb = Vp;
mb = pmass;
bspeed = 0.74 * (2 * mb * vb)/(ma + mb);

%initial bottle deceleration due to friction
a = -1 * (Ff) / bmass;

%calculate time to stop if theoretically always on ramp
timeToStop = -1 * bspeed / a;
theoreticalDistance = 0.5 * a * timeToStop^2 + bspeed * timeToStop;

%check to see if bottle will stop before falling off of ramp
if theoreticalDistance <= rlength %doesn't fall off of ramp
    distance = theoreticalDistance;
else
    %% place code for off-ramp here
    t1 = (bspeed + (bspeed^2 - 2 * a * -1 * rlength)^(1/2))/(a);
    t2 = (bspeed - (bspeed^2 - 2 * a * -1 * rlength)^(1/2))/(a);
    tvect = [t1 t2]; %store time values in an array
    t = min(abs(tvect)); %pull most realistic time
    V = a * t + bspeed;
    distanceT = V * fallTime;
    distance = rlength + distanceT;
end

if mindist < distance
    if distance < maxdist
        landed = 1;
    else
        landed = 0;
    end
else
    landed = 0;
end

end
