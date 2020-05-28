%%pSim2

%% Physical parameter initialization

clear; clc; clf; close;

%gravity
g = 9.8;

%pendulum
pmass = 0.7704;
plength = 1.97;

%bottle
bmass = 0.1494;
blength = 0.21;
bwidth = 0.092;

%ramp
ramp_rem = 0.67;
%measure from middle of bottle
rlength = ramp_rem + blength/2;
rangle = 10; %ramp angle in degrees for static friction

%height metrics
rheight = 0.915;
buckheight = 0.372;
buckwidth = 0.237;
marginError = buckwidth/2 - blength/2;
trueheight = rheight - buckheight;
buckdist = 1.675;
fulldist = buckdist + rlength + buckwidth/2 - blength/2;
mindist = buckdist + rlength - blength/2;
maxdist = buckdist + rlength + buckwidth - blength/2;

%% calc time for bottle to fall
%assume bucket is set in invisible floor for ease
fallTime = (2 * g * trueheight)^(1/2)/(g);


%% height gather

h = input('From what height is the pendulum dropped? ');


%% Basic physical factor calc

%friction on bottle
%following mu-value calculation derived from static friction calc equation
%for network batch generation efficiency compressed to 1 operation
mu = tand(rangle);
Fn = bmass * g;
Ff = mu * Fn;


%simple pendulum velocity
PEp = pmass * g * h;
simpleKEp = PEp;
%calculate velocity of pendulum
%use v^2=2gh
simpleVp = (2 * g * h)^(1/2);


%assume 0 air resistance
Fa = 0;

%air resistance
muA = 1;

%factor in air resistance to pendulum velocity for elasticity calc
Vp = simpleVp * muA;

%elastic collision math where va,ma are bottle and vb,mb are pendulum
va = 0;
ma = bmass;
vb = Vp;
mb = pmass;
bspeed = 0.74 * ((2 * mb * vb)/(ma + mb) + va*(ma - mb)/(ma + mb));

%initial bottle deceleration due to friction
a = -1 * (Ff + Fa) / bmass;

%calculate time to stop if theoretically always on ramp
timeToStop = -1 * bspeed / a;
theoreticalDistance = 0.5 * a * timeToStop^2 + bspeed * timeToStop;

%check to see if bottle will stop before falling off of ramp
if theoreticalDistance <= rlength %doesn't fall off of ramp
    finalPosition = theoreticalDistance;
    disp(['Final distance from launch point: ',num2str(finalPosition),' m']);
    disp(['Effective ramp length: ',num2str(rlength),' m']);
else
    %% place code for off-ramp here
    t1 = (bspeed + (bspeed^2 - 2 * a * -1 * rlength)^(1/2))/(a);
    t2 = (bspeed - (bspeed^2 - 2 * a * -1 * rlength)^(1/2))/(a);
    tvect = [t1 t2]; %store time values in an array
    t = min(abs(tvect)); %pull most realistic time
    V = a * t + bspeed;
    distance = V * fallTime;
    finalPosition = rlength + distance;
end

if mindist < finalPosition
    if finalPosition < maxdist
        disp('Landed in the bucket!  Way to go!');
    else
        disp(['Overshot by ',num2str(finalPosition-maxdist),' m']);
    end
else
    disp(['Undershot by ',num2str(mindist-finalPosition),' m']);
end
