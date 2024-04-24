clear
clc
%%
load data.mat % time-series
load maindata_Carlstrom.mat % estimates from the paper
%%

% bounds
lb = [0.1 0.14 0.05 0.0048 0.005 0.21 0.05 0.01 0.367879441171442 0.9 1 0.5 0.5 5 1 1 0.03 0.75 0.1 0.8 0.1 0.93 0.95 0.3 0.7 0.95 0.5 0.5 0.8 0 0.2 0.01 0.1 0.21 0.32 0.001 0.001 0.4 0.4 0.01];
ub = [0.5 0.78 0.78 0.0056 0.0057 0.98 0.25 0.1 1.5 0.9999 10 0.9 0.7 10 2.5 10 0.21 0.95 0.7 1 0.7 0.99 0.99 0.7 0.99 0.999 0.9 0.9 0.99 3 2 1 1 1 1 0.05 0.2 1.5 10 4];

ff     = @(param)minusloglikelihood(param,data); % function to minimise
x0     = prpr; % initialisation for fmincon
x0(20) = 0.99; % instead of equal to one
x0(40) =  0.1; % instead of equal to one

AA  = []; % no linear inequality constraints
bb  = []; % no linear inequality constraints
Aeq = []; % no linear equality constraints
beq = []; % no linear equality constraints
non = []; % for nonlinear constraints

options = optimoptions('fmincon','Display','iter','Algorithm','interior-point');
options = optimoptions(options,'MaxFunctionEvaluations',10000);

%%

x = fmincon(ff,x0.',AA,bb,Aeq,beq,lb,ub,non,options) 

[AA,~,~,~,~] = Carlstrom_solve(x,1);
C = zeros(5,37);
C(1,10) = 1; % 1 - federal funds rate (FFR)
C(1,15) = 1; % 2 - private investment (FPI)
C(2,7)  = 1; % 3 - GDPDEF deflator-based inflation (GDPDEF)
C(3,5)  = 1; % 4 - WAGE wages (WAGE)
C(4,9)  = 1; % 5 - CONS consumption (CONS)
[like,datalast] = minusloglikelihood_add(x,data);

frcsts = zeros(37,5);
frcsts(:,1) = AA*datalast;
frcsts(:,2) = AA*frcsts(:,1);
frcsts(:,3) = AA*frcsts(:,2);
frcsts(:,4) = AA*frcsts(:,3);
frcsts(:,5) = AA*frcsts(:,4);

obsvfrcst = zeros(5,5);
for j = 1:5
    obsvfrcst(j,:) = C*frcsts(:,j);
end