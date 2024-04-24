function [modlambdaC, modlambdaF]=BP(theta0)
%calculates the eigenvalues of C and F to determine the uniqueness of the
%solution using BP-method

% set parameter values

alpha           =  theta0(1);
iotap           =  theta0(2);
iotaw           =  theta0(3);
gammaz          =  theta0(4); 
gammau          =  theta0(5);
h               =  theta0(6);
lambdapss       =  theta0(7);
lambdawss       =  theta0(8);
Lss             =  theta0(9);
beta            = theta0(10);
Psi             = theta0(11);
xip             = theta0(12);
xiw             = theta0(13);
theta           = theta0(14);
Sadj            = theta0(15);
phipi           = theta0(16);
phidx           = theta0(17);
rhoR            = theta0(18);
hik             = theta0(30);


% calibrated data

delta   = 0.025;
gss     = 1/(1-0.22);
gamma   = 0.94;
rp      = 0.005;
kappa   = 1.95;
thetag  = 0.95;
nu      = 0.19;
cb      = 0;

expLss=exp(Lss);
Rkss=exp(gammaz + gammau)/beta-1+delta;
sss=1/(1+lambdapss);
wss=(sss*((1-alpha)^(1-alpha))/((alpha^(-alpha))*Rkss^alpha))^(1/(1-alpha));
kLss=(wss/Rkss)*alpha/(1-alpha);
FLss=(kLss^alpha-Rkss*kLss-wss);
yLss=kLss^alpha-FLss;
kss=kLss*expLss;
iss=(1-(1-delta)*exp(-gammaz-gammau))*kss*exp(gammaz+gammau);
F=FLss*expLss;
yss=yLss*expLss;
css=yss/gss-iss;

expg=exp(gammaz);
expgmiu=exp(gammaz + gammau);
kappaw=((1-xiw*beta)*(1-xiw))/(xiw*(1+beta)*(1+Psi*(1+1/lambdawss)));

% compute matrices for first-order model
% M00 * y[t] = M10 * y[t-1] + M01 * E_t(y[t+1]) + wtilde[t]
M00 = zeros(21,21);
M10 = zeros(21,21);
M01 = zeros(21,21);

M00(1,1) = 1;
M00(1,2) = -(((yss+F)/yss))*alpha;
M00(1,3) = -(((yss+F)/yss)*(1-alpha));
M00(2,4) = 1;
M00(2,5) = -1;
M00(2,3) = -1;
M00(2,2) = 1;
M00(3,6) = 1;
M00(3,4) = -alpha;
M00(3,5) = -(1-alpha);
M00(4,7) = 1;
M00(4,6) = -(((1-beta*xip)*(1-xip)/((1+iotap*beta)*xip)));
M00(5,8) = ((expg-h*beta)*(expg-h));
M00(5,9) = ((expg^2+beta*h^2));
M00(6,8) = 1;
M00(6,10) = -1;
M00(7,4) = 1;
M00(7,11) = -theta;
M00(8,8) = -1;
M00(8,12) = -nu;
M00(8,2) = -nu;
M00(8,13) = nu;
M00(8,21) = 1;
M00(9,12) = 1;
M00(9,15) = -(1+beta)*(Sadj*expgmiu^2);
M00(10,2) = 1;
M00(10,11) = -1;

M00(11,16) = 1;
M00(11,15) = -((1-(1-delta)*exp(-gammaz-gammau)));
M00(12,5) = 1;
M00(12,7) = ((1+beta*iotaw)/(1+beta));
M00(12,17) = kappaw;
M00(13,17) = 1;
M00(13,5) = -1;
M00(13,3) = Psi;
M00(13,8) = -1;
M00(14,10) = 1;
M00(14,7) = -((1-rhoR)*phipi);
M00(14,18) = -((1-rhoR)*phidx);
M00(15,18) = 1;
M00(15,1) = -1;
M00(15,11) = (kss*Rkss/yss);
M00(16,1) = 1/gss;
M00(16,9) = -css/yss;
M00(16,15) = -iss/yss;
M00(16,11) = -(kss*Rkss/yss);
M00(17,19) = 1;
M00(17,10) = -1;
M00(18,14) = 1;
M00(18,12) = -(beta*exp(-gammaz-gammau)*(1-delta));
M00(18,4) = -((1-beta*exp(-gammaz-gammau)*(1-delta)));
M00(19,13) = 1;
M00(19,20) = -(1-kappa)*(gamma/beta);
M00(19,14) = -((1+rp)*gamma*kappa)/beta;
M00(20,20) = 1;
M00(20,14) = -(1 + thetag*(hik - 1));
M00(21,21) = 1;


M10(4,7) = (iotap/(1+iotap*beta));
M10(5,9) = (expg*h);
M10(9,15) = -(Sadj*expgmiu^2);
M10(10,16) = -1;
M10(11,16) = ((1-delta)*exp(-gammaz-gammau));
M10(12,5) = (1/(1+beta));
M10(12,7) = (iotaw/(1+beta));
M10(14,10) = rhoR;
M10(14,18) = -((1-rhoR)*phidx);
M10(18,12) = -1;
M10(19,13) = (gamma/beta);
M10(19,16) = gamma*kappa*(rp/beta);%changed to kbar
M10(19,12) = gamma*kappa*(rp/beta);
M10(20,19) = 1;
M10(20,21) = -(1 + thetag*(hik - 1));


M01(4,7) = (beta/(1+iotap*beta));
M01(5,9) = (beta*h*expg);
M01(6,8) = 1; 
M01(6,7) = -1; 
M01(8,8) = -1;
M01(9,15) = -beta*(Sadj*expgmiu^2);
M01(12,5) = (beta/(1+beta));
M01(12,7) = (beta/(1+beta));
M01(17,7) = -1;
M01(21,14) = 1;
M01(21,21) = 1;


A = inv(M00)*M10;
B = inv(M00)*M01;

[dim1,dim2] = size(A);

% compute 'C'-matrix
C = eye(dim1);         % initial condition
F = eye(dim1);         % initial condition
eps1 = 10^(-15);        % convergence criterion for C
eps2 = 10^(-15);        % convergence criterion for F
crit1 = 1; crit2 = 1;  % initial conditions
iter = 0;
while crit1 >= eps1 || crit2 >= eps2
   Ci = inv(eye(dim1)-B*C)*A;
   Fi = inv(eye(dim1)-B*C)*B;
   crit1 = max(max(abs(Ci-C))); crit2 = max(max(abs(Fi-F)));
   C = Ci; F = Fi;
   iter = iter+1;
   if iter > 1000, break, end
end

% display of Results
%display(C)
lambdaC = eig(C);
%display(lambdaC)
modlambdaC = abs(eig(C));
%display(modlambdaC)
%display(F)
lambdaF = eig(F);
%display(lambdaF)
modlambdaF = abs(eig(F));
%display(modlambdaF)

end 