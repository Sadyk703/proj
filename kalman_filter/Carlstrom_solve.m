%******************************************************************************
%*      Solve LS (2004) new Keynesian Monetary Policy DSGE using GENSYS
%*      Translated from the Gauss code of Lubik and Schorfheide.
%*      
%**************************************************************************/
function [T1,TC,T0,TETA,RC] = Carlstrom_solve(para,msel)

%para = prpr;

alpha      =  para(1);
iotap      =  para(2);
iotaw      =  para(3);
gammaz     =  para(4); 
gammau     =  para(5);
h          =  para(6);
lambdapss  =  para(7);
lambdawss  =  para(8);
Lss        =  para(9);
beta       = para(10);
Psi        = para(11);
xip        = para(12);
xiw        = para(13);
theta      = para(14);
Sadj       = para(15);
phipi      = para(16);
phidx      = para(17);
rhoR       = para(18);
rhoz       = para(19);
rhog       = para(20);
rhou       = para(21);
rhop       = para(22);
rhow       = para(23);
rhob       = para(24);
thetap     = para(25);
thetaw     = para(26);
rhosigma   = para(27);
rhonw      = para(28);
rhomu      = para(29);
hik        = para(30);
sigmaz     = para(31);
sigmag     = para(32);
sigmau     = para(33);
sigmarho   = para(34);
sigmaw     = para(35);
sigmab     = para(36);
sigmasigma = para(37);
sigmanw    = para(38);
sigmamu    = para(39);
sigmamp    = para(40);

%% calibrated data

delta   = 0.025;
gss     = 1/(1-0.22);
gamma   =  0.94;
rp      = 0.005;
kappa   =  1.95;
thetag  =  0.95;
nu      =  0.19;
cb      =     0;

%% variable indices

v_y        =  1;
v_k        =  2;
v_L        =  3;
v_rho      =  4;
v_w        =  5;
v_s        =  6;
v_pi       =  7;
v_lambda   =  8;
v_c        =  9;
v_R        = 10;
v_u        = 11;
v_q        = 12;
v_n        = 13;
v_rk       = 14;
v_i        = 15;
v_kbar     = 16;
v_gw       = 17;
v_x        = 18;
v_rd       = 19;
v_rl       = 20;
v_Erk      = 21;
v_Epi      = 22;
v_Ec       = 23;
v_Elambda  = 24;
v_Ei       = 25;
v_Ew       = 26;
v_sigma    = 27; %idiosyncratic shock ?t
v_lambda_p = 28; %price markup shocks
v_z        = 29; %TFP shock
v_lambda_w = 30; %wage markup shocks
v_ups      = 31; %productivity shock
v_mu       = 32; % investment shock
v_b        = 33; %intertemporal preference shock
v_eta_nw   = 34;
v_g        = 35; %government spending shock
v_elmbdaw  = 36;
v_elmbdap  = 37;

 
%% shock indices (neps)
e_sigma        =  1;
e_lambda_p     =  2;
e_z            =  3;
e_lambda_w     =  4;
e_ups          =  5;
e_mu           =  6;
e_b            =  7;
e_eta_nw       =  8;
e_g            =  9;
e_mp           = 10;

%% expectation error indices (neta)
n_rk     = 1;
n_pi     = 2;
n_c      = 3;
n_lambda = 4;
n_i      = 5;
n_w      = 6;

%% summary
neq  = 37;
neps = 10;
neta =  6;

%% initialize matrices
GAM0 = zeros(neq, neq);
GAM1 = zeros(neq, neq);
   C = zeros(neq,   1);
 PSI = zeros(neq,neps);
 PPI = zeros(neq,neta);
 
 %% Equations
 % equation 1
GAM0(1,v_y) = 1;
GAM0(1,v_k) = -(ysteadystate(para)+fsteadystate(para))/ysteadystate(para)*alpha;
GAM0(1,v_L) = -(ysteadystate(para)+fsteadystate(para))/ysteadystate(para)*(1-alpha);
% equation 2
GAM0(2,v_rho) =  1;
GAM0(2,v_w)   = -1;
GAM0(2,v_L)   = -1;
GAM0(2,v_k)   =  1;
% equation 3
GAM0(3,v_s)   =          1;
GAM0(3,v_rho) =     -alpha;
GAM0(3,v_w)   = -(1-alpha);
% equation 4
GAM0(4,v_pi)       =  1;
GAM0(4,v_s)        = -(1-beta*xip)*(1-xip)/((1+beta*iotap)*xip);
GAM0(4,v_Epi)      = -beta/(1+beta*iotap);
GAM0(4,v_lambda_p) = -1;
GAM1(4,v_pi)       = iotap/(1+beta*iotap);
% equation 5
GAM0(5,v_lambda) = (exp(gammaz)-h*beta)*(exp(gammaz)-h);
GAM0(5,v_c)      = exp(2*gammaz)+h^2*beta;
GAM0(5,v_Ec)     = -h*beta*exp(gammaz);
GAM0(5,v_z)      = -h*beta*exp(gammaz)*rhoz+h*exp(gammaz);
GAM0(5,v_ups)    = -(h*beta*exp(gammaz)*rhou-h*exp(gammaz))*alpha/(1-alpha);
GAM0(5,v_b)      = -(exp(gammaz)*h+exp(2*gammaz)+beta*h^2)/(1-rhob);% (exp(gammaz)-h*beta*rhob)*(exp(gammaz)-h);
GAM1(5,v_c)      = h*exp(gammaz);
% equation 6
GAM0(6,v_lambda)  =    1;
GAM0(6,v_R)       =   -1;
GAM0(6,v_Elambda) =   -1;
GAM0(6,v_Epi)     =    1;
GAM0(6,v_z)       = rhoz;
GAM0(6,v_ups)     = rhou*alpha/(1-alpha);
% equation 7
GAM0(7,v_rho) =      1;
GAM0(7,v_u)   = -theta;
% equation 8
GAM0(8,v_lambda)  =    1;
GAM0(8,v_q)       =   nu;
GAM0(8,v_k)       =   nu;
GAM0(8,v_n)       =  -nu;
GAM0(8,v_Elambda) =   -1;
GAM0(8,v_Erk)     =   -1;
GAM0(8,v_z)       = rhoz;
GAM0(8,v_ups)     = rhou*alpha/(1-alpha);
GAM0(8,v_sigma)   = 1;
% equation 9
GAM0(9,v_q)   = 1;
GAM0(9,v_i)   = -exp(2*(gammaz+gammau))*Sadj*(1+beta);
GAM0(9,v_Ei)  = beta*exp(2*(gammaz+gammau))*Sadj;
GAM0(9,v_mu)  = 1;
GAM0(9,v_z)   = -exp(2*(gammaz+gammau))*Sadj*(1-beta*rhoz);
GAM0(9,v_ups) = -exp(2*(gammaz+gammau))*Sadj*(1-beta*rhou)*1/(1-alpha);
GAM1(9,v_i)   = -exp(2*(gammaz+gammau))*Sadj;
% equation 10
GAM0(10,v_k)    =  1;
GAM0(10,v_u)    = -1;
GAM0(10,v_z)    =  1;
GAM0(10,v_ups)  = 1/(1-alpha);
GAM1(10,v_kbar) =  1;
% equation 11
GAM0(11,v_kbar) = 1;
GAM0(11,v_i)    = -(1-(1-delta)*exp(-gammaz-gammau));
GAM0(11,v_mu)   = -(1-(1-delta)*exp(-gammaz-gammau));
GAM0(11,v_z)    = (1-delta)*exp(-gammaz-gammau);
GAM0(11,v_ups)  = (1-delta)*exp(-gammaz-gammau)/(1-alpha);
GAM1(11,v_kbar) = (1-delta)*exp(-gammaz-gammau);
% equation 12
GAM0(12,v_w)        =  1;
GAM0(12,v_gw)       = (1-xiw*beta)*(1-xiw)/(xiw*(1+beta)*(1+Psi*(1+1/lambdawss)));
GAM0(12,v_pi)       = (1+beta*iotaw)/(1+beta);
GAM0(12,v_Ew)       = -beta/(1+beta);
GAM0(12,v_Epi)      = -beta/(1+beta);
GAM0(12,v_z)        = (1+beta*iotaw-beta*rhoz)/(1+beta);
GAM0(12,v_ups)      = (1+beta*iotaw-beta*rhou)/(1+beta)*alpha/(1-alpha);
GAM0(12,v_lambda_w) = -1;
GAM1(12,v_w)        = 1/(1+beta);
GAM1(12,v_pi)       = iotaw/(1+beta);
GAM1(12,v_z)        = iotaw/(1+beta);
GAM1(12,v_ups)      = iotaw/(1+beta)*alpha/(1-alpha);
% eqiation 13
GAM0(13,v_gw)     =   1;
GAM0(13,v_w)      =  -1;
GAM0(13,v_lambda) =  -1;
GAM0(13,v_L)      = Psi;
GAM0(13,v_b)      =   1/((1-rhob)*(exp(gammaz)-h*beta*rhob)*(exp(gammaz)-h)/(exp(gammaz)*h+exp(2*gammaz)+beta*h^2));
% equation 14
GAM0(14,v_R)  =    1;
GAM0(14,v_pi) = -(1-rhoR)*phipi;
GAM0(14,v_x)  = -(1-rhoR)*phidx;
GAM1(14,v_R)  = rhoR;
GAM1(14,v_x)  = -(1-rhoR)*phidx;
PSI(14,e_mp)  =    1;
% equation 15
GAM0(15,v_x) = 1;
GAM0(15,v_y) = -1;
GAM0(15,v_u) = rhosteadystate(para)*ksteadystate(para)/ysteadystate(para);
% equation 16
GAM0(16,v_y) = 1/gss;
GAM0(16,v_c) = -csteadystate(para)/ysteadystate(para);
GAM0(16,v_i) = -isteadystate(para)/ysteadystate(para);
GAM0(16,v_u) = -rhosteadystate(para)*ksteadystate(para)/ysteadystate(para);
GAM0(16,v_g) = -1/gss;
% equation 17
GAM0(17,v_rd)  =  1;
GAM0(17,v_R)   = -1;
GAM0(17,v_Epi) =  1;
% equation 18
GAM0(18,v_rk)  =  1;
GAM0(18,v_q)   = -beta*(1-delta)*exp(-gammaz-gammau);
GAM0(18,v_rho) = -(1-beta*(1-delta)*exp(-gammaz-gammau));
GAM1(18,v_q)   = -1;
% equation 19
GAM0(19,v_n)      = 1;
GAM0(19,v_rk)     = -kappa*gamma/beta*(1+rp);
GAM0(19,v_rl)     = gamma/beta*(kappa-1);
GAM0(19,v_z)      = 1;
GAM0(19,v_ups)    = 1/(1-alpha);
GAM0(19,v_eta_nw) = -1;
GAM1(19,v_n)      = gamma/beta;
GAM1(19,v_kbar)   = gamma*kappa*rp/beta;
GAM1(19,v_q)      = gamma*kappa*rp/beta;
% equation 20
GAM0(20,v_rl)  = 1;
GAM0(20,v_rk)  = -(1+thetag*(hik-1));
GAM1(20,v_rd)  = 1;
GAM1(20,v_Erk) = -(1+thetag*(hik-1));
% equation 21
GAM0(21,v_eta_nw) = 1;
GAM1(21,v_eta_nw) = rhonw;
PSI(21,e_eta_nw)  = 1;
% equation 22
GAM0(22,v_sigma) = 1;
GAM1(22,v_sigma) = rhosigma;
PSI(22,e_sigma)  = 1;
% equation 23
GAM0(23,v_lambda_p)    =  1;
GAM0(23,v_elmbdap)     = -1;
GAM1(23,v_lambda_p)    = rhop;
GAM1(23,v_elmbdap)     = -thetap;
% equation 36
GAM0(36,v_elmbdap)     =  1;
PSI(36,e_lambda_p)     =  1;
% equation 24
GAM0(24,v_z) = 1;
GAM1(24,v_z) = rhoz;
PSI(24,e_z)  = 1;
% equation 25
GAM0(25,v_lambda_w)    =  1;
GAM0(25,v_elmbdaw)     = -1;
GAM1(25,v_lambda_w)    = rhow;
GAM1(25,v_elmbdaw)     = -thetaw;
% equation 37
GAM0(37,v_elmbdaw)     =  1;
PSI(37,e_lambda_w)     =  1;
% equation 26
GAM0(26,v_ups) = 1;
GAM1(26,v_ups) = rhou;
PSI(26,e_ups)  = 1;
% equation 27
GAM0(27,v_mu) = 1;
GAM1(27,v_mu) = rhomu;
PSI(27,e_mu)  = 1;
% equation 28
GAM0(28,v_b) = 1;
GAM1(28,v_b) = rhob;
PSI(28,e_b)  = 1;
% equation 29
GAM0(29,v_g) = 1;
GAM1(29,v_g) = rhog;
PSI(29,e_g)  = 1;
% equation 30
GAM0(30,v_rk)  = 1;
GAM1(30,v_Erk) = 1;
PPI(30,n_rk)   = 1;
% equation 31
GAM0(31,v_pi)  = 1;
GAM1(31,v_Epi) = 1;
PPI(31,n_pi)   = 1;
% equation 32
GAM0(32,v_c)  = 1;
GAM1(32,v_Ec) = 1;
PPI(32,n_c)   = 1;
% equation 33
GAM0(33,v_lambda)  = 1;
GAM1(33,v_Elambda) = 1;
PPI(33,n_lambda)   = 1;
% equation 34
GAM0(34,v_i)  = 1;
GAM1(34,v_Ei) = 1;
PPI(34,n_i)   = 1;
% equation 35
GAM0(35,v_w)  = 1;
GAM1(35,v_Ew) = 1;
PPI(35,n_w)   = 1;

[T1,TC,T0,TY,M,TZ,TETA,GEV,RC]=gensys_mod(GAM0, GAM1, C, PSI, PPI, 1); %use Sims-based code, also uses ordqz
  

end
