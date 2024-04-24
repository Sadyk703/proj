%clear
%clc
%%

load maindata

alpha           =  prpr(1);
iotap           =  prpr(2);
iotaw           =  prpr(3);
gammaz          =  prpr(4); 
gammau          =  prpr(5);
h               =  prpr(6);
lambdapss       =  prpr(7);
lambdawss       =  prpr(8);
Lss             =  prpr(9);
beta            = prpr(10);
Psi             = prpr(11);
xip             = prpr(12);
xiw             = prpr(13);
theta           = prpr(14);
Sadj            = prpr(15);
phipi           = prpr(16);
phidx           = prpr(17);
rhoR            = prpr(18);
rhoz            = prpr(19);
rhog            = prpr(20);
rhou            = prpr(21);
rhop            = prpr(22);
rhow            = prpr(23);
rhob            = prpr(24);
thetap          = prpr(25);
thetaw          = prpr(26);
rhosigma        = prpr(27);
rhonw           = prpr(28);
rhomu           = prpr(29);
hik             = prpr(30);
sigmaz          = prpr(31);
sigmag          = prpr(32);
sigmau          = prpr(33);
sigmarho        = prpr(34);
sigmaw          = prpr(35);
sigmab          = prpr(36);
sigmasigma      = prpr(37);
sigmanw         = prpr(38);
sigmamu         = prpr(39);
sigmamp         = prpr(40);

%% calibrated data

delta   = 0.025;
gss     = 1/(1-0.22);
gamma   = 0.94;
rp      = 0.005;
kappa   = 1.95;
thetag  = 0.95;
nu      = 0.19;
cb      = 0;

%%
%rss=exp(gammaz)/beta-1;
%rss100=rss*100;
%pss=pss100/100;

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


%%

%********************************************************************
%**      matrices of canonical system
%********************************************************************

%** variable indices **
v_y   = 1;
v_k   = 2;
v_L   = 3;
v_rho = 4;
v_w   = 5;
v_s   = 6;
v_pi  = 7;
v_lambda   = 8;
v_c   = 9;
v_R   = 10;
v_u   = 11;
v_q   = 12;
v_n   = 13;
v_rk   = 14;
v_i   = 15;
v_kbar   = 16;
v_gw   = 17;
v_x   = 18;
v_rd   = 19;
v_rl   = 20;
v_Erk   = 21;
v_Epi   = 22;
v_Ec   = 23;
v_Elambda   = 24;
v_Ei   = 25;
v_Ew   = 26;
v_sigma   = 27; %idiosyncratic shock ?t
v_lambda_p   = 28; %price markup shocks
v_z   = 29; %TFP shock
v_lambda_w   = 30; %wage markup shocks
v_v   = 31; %productivity shock
v_mu  = 32; % investment shock
v_b  = 33; %intertemporal preference shock
v_eta_nw   = 34; %Net worth shock
%v_eta_mp  = 34; %monetary policy shock
v_g  = 35; %government spending shock
v_elmbdaw  = 36;
v_elmbdap  = 37;

%** shock indices **
e_sigma =1;
e_lambda_p =2;
%e_lambda_p_t_1 =3;
e_z  = 3;
e_lambda_w  = 4;
%e_lambda_w_t_1  = 6;
e_v  = 5;
e_mu  = 6;
e_b  = 7;
e_eta_nw = 8;
e_g = 9;
e_mp = 10;


%** expectation error indices **
n_rk   = 1;
n_pi   = 2;
n_c   = 3;
n_lambda   = 4;
n_i   = 5;
n_w   = 6;

%%

%** initialize matrices **
GAM0 = zeros(37,37);
GAM1 = zeros(37,37);
   C = zeros(37,1);
 PSI = zeros(37,10);
 PPI = zeros(37,6);
%% 
%**********************************************************
%**      1. Production Function
%**********************************************************
GAM0(1,v_y)   = 1;
GAM0(1,v_k)   = -(((yss+F)/yss))*alpha;
GAM0(1,v_L)   = -(((yss+F)/yss)*(1-alpha));
 
%**********************************************************
%**      2. cost minimization
%**********************************************************
GAM0(2,v_rho)   = 1;
GAM0(2,v_w)   = -1;
GAM0(2,v_L)   = -1;
GAM0(2,v_k)   = 1;

%**********************************************************
%**      3. marginal cost
%**********************************************************
GAM0(3,v_s)   = 1;
GAM0(3,v_rho)   = -alpha;
GAM0(3,v_w)   = -(1-alpha);

%**********************************************************
%**      4. Phillips curve
%**********************************************************
GAM0(4,v_pi)   = 1;
GAM0(4,v_Epi)  = -(beta/(1+iotap*beta));
GAM0(4,v_s)   = -(((1-beta*xip)*(1-xip)/((1+iotap*beta)*xip)));
GAM0(4,v_lambda_p) = -1;

GAM1(4,v_pi) = (iotap/(1+iotap*beta));
%%
%**********************************************************
%**      5. Consumption FOC
%**********************************************************
GAM0(5,v_lambda)   = ((expg-h*beta)*(expg-h));
GAM0(5,v_Ec)   = - (beta*h*expg);
GAM0(5,v_c)   =  ((expg^2+beta*h^2));
GAM0(5,v_z)   =  -((beta*h*expg*rhoz-h*expg));
GAM0(5,v_b) =  - ((expg-h*beta*rhob)*(expg-h)/((1-rhob)*(expg-h*beta*rhob)*(expg-h)/(expg*h+expg^2+beta*h^2)));
GAM0(5,v_v) = - ((beta*h*expg*rhou-h*expg)*alpha/(1-alpha));

GAM1(5,v_c)   =  expg*h;
%**********************************************************
%**      6. Euler equation
%**********************************************************
GAM0(6,v_lambda)   = 1;
GAM0(6,v_R)   = -1;
GAM0(6,v_Elambda)   = -1;
GAM0(6,v_Epi)   = 1;
GAM0(6,v_Epi)   = 1;
GAM0(6,v_z)   = rhoz;
GAM0(6,v_v)   = (rhou*alpha/(1-alpha));

%**********************************************************
%**      7. Capital Utilization
%**********************************************************
GAM0(7,v_rho)   = 1;
GAM0(7,v_u)   = -theta;

%**********************************************************
%**      8. Capital Utilization
%**********************************************************
GAM0(8,v_Erk)   = 1;
GAM0(8,v_lambda)   = -1;
GAM0(8,v_Elambda)   = 1;
GAM0(8,v_z)   = -rhoz;
GAM0(8,v_v)   = -(rhou*alpha/(1-alpha));
GAM0(8,v_q)   = -nu;
GAM0(8,v_k)   = -nu;
GAM0(8,v_n)   = nu;
GAM0(8,v_sigma)   = -1;

%%
%**********************************************************
%**      9. Investment FOC
%**********************************************************
GAM0(9,v_q)   = 1;
GAM0(9,v_mu)   = 1;
GAM0(9,v_i)   = -(1+beta)*(Sadj*expgmiu^2);
GAM0(9,v_Ei)   = (beta*(Sadj*expgmiu^2));
GAM0(9,v_z)   = -(1-beta*rhoz)*(Sadj*expgmiu^2);
GAM0(9,v_v)   = -(((1-beta*rhou)*(Sadj*expgmiu^2))*(1/(1-alpha)));

GAM1(9,v_i)   = -(Sadj*expgmiu^2);
%**********************************************************
%**      10. Capital Input
%**********************************************************
GAM0(10,v_k)   = 1;
GAM0(10,v_u)   = -1;
GAM0(10,v_z)   = 1;
GAM0(10,v_v)   = (1/(1-alpha));

GAM1(10,v_kbar)   = 1;

%**********************************************************
%**      11. Capital Accumulation
%**********************************************************
GAM0(11,v_kbar)   = 1;
GAM0(11,v_z)   = ((1-delta)*exp(-gammaz-gammau));
GAM0(11,v_v)   = (((1-delta)*exp(-gammaz-gammau))*(alpha/(1-alpha)+1));
GAM0(11,v_mu)   = -((1-(1-delta)*exp(-gammaz-gammau)));
GAM0(11,v_i)   = -((1-(1-delta)*exp(-gammaz-gammau)));

GAM1(11,v_kbar)   = ((1-delta)*exp(-gammaz-gammau));

%**********************************************************
%**      12. Wage Phillips Curve
%**********************************************************
GAM0(12,v_w)   = 1;
GAM0(12,v_Ew)   = -(beta/(1+beta));
GAM0(12,v_gw)   = (kappaw);
GAM0(12,v_pi)   = ((1+beta*iotaw)/(1+beta));
GAM0(12,v_Epi)   = -(beta/(1+beta));
GAM0(12,v_z)   = ((1+beta*iotaw-beta*rhoz)/(1+beta));
GAM0(12,v_v)   = ((1+beta*iotaw-beta*rhou)/(1+beta))*(alpha/(1-alpha));
GAM0(12,v_lambda_w)   = -1;

GAM1(12,v_w)   = (1/(1+beta));
GAM1(12,v_pi)   = (iotaw/(1+beta));
GAM1(12,v_z)   = (iotaw/(1+beta));
GAM1(12,v_v)   = (iotaw/(1+beta))*(alpha/(1-alpha));

%%
%**********************************************************
%**      13. Wage Gap
%**********************************************************
GAM0(13,v_gw)   = 1;
GAM0(13,v_w)   = -1;
GAM0(13,v_L)   = Psi;
GAM0(13,v_b)   = (1/((1-rhob)*(expg-h*beta*rhob)*(expg-h)/(expg*h+expg^2+beta*h^2)));
GAM0(13,v_lambda)   = 1;

%**********************************************************
%**      14. Monetary Policy Rule
%**********************************************************
GAM0(14,v_R) = 1;
GAM0(14,v_pi) = -((1-rhoR)*phipi);
GAM0(14,v_x) = -((1-rhoR)*phidx);


GAM1(14,v_R) = rhoR;
GAM1(14,v_x) = -((1-rhoR)*phidx);
PSI(14,e_mp)  =  1;

%**********************************************************
%**      15. GDP
%**********************************************************
GAM0(15,v_x) = 1;
GAM0(15,v_y) = -1;
GAM0(15,v_u) = (kss*Rkss/yss);

%**********************************************************
%**      16.Market Clearing
%**********************************************************
GAM0(16,v_y) = 1/gss;
GAM0(16,v_c) = -css/yss;
GAM0(16,v_i) = -iss/yss;
GAM0(16,v_u) = -(kss*Rkss/yss);

%%
%**********************************************************
%**      17. Definition of risk-free rate from Fisher equation
%**********************************************************
GAM0(17,v_rd) = 1;
GAM0(17,v_R) = -1;
GAM0(17,v_Epi) = 1;

%**********************************************************
%**      18. Definition of Realized Return 
%**********************************************************
GAM0(18,v_rk) = 1;
GAM0(18,v_q) = -(beta*exp(-gammaz-gammau)*(1-delta));
GAM0(18,v_rho) = -((1-beta*exp(-gammaz-gammau)*(1-delta)));

GAM1(18,v_q) = -1;

%**********************************************************
%**      19. Evolution of Net worth
%**********************************************************
GAM0(19,v_n) = 1;
GAM0(19,v_rl) = -(1-kappa)*(gamma/beta);
GAM0(19,v_rk) = -((1+rp)*gamma*kappa)/beta;
GAM0(19,v_z) = 1;
GAM0(19,v_v) = (1/(1-alpha));
GAM0(19,v_eta_nw) = -1;

GAM1(19,v_n) = (gamma/beta);
GAM1(19,v_kbar) = gamma*kappa*(rp/beta);
GAM1(19,v_q) = gamma*kappa*(rp/beta);

%**********************************************************
%**      20. Lender's return
%**********************************************************
GAM0(20,v_rl) = 1;
GAM0(20,v_rk) = -(1 + thetag*(hik - 1));

GAM1(20,v_rd) = 1;
GAM1(20,v_Erk) = -(1 + thetag*(hik - 1));
%%

%**********************************************************
%**      21. Erk(+1)
%**********************************************************
%GAM0(21,v_Erk) = 1;


%**********************************************************
%**      Shock process
%**********************************************************
%** 21 v_eta_nw ** Net worth shock
GAM0(21,v_eta_nw) = 1;
GAM1(21,v_eta_nw) = rhonw;
 PSI(21,e_eta_nw) = 1;


%** 22 v_sigma ** idiosyncratic shock
GAM0(22,v_sigma) = 1;
GAM1(22,v_sigma) = rhosigma;
 PSI(22,e_sigma) = 1; 
 
%** 23 v_lambda_p ** 
GAM0(23,v_lambda_p)    =  1;
GAM0(23,v_elmbdap)     = -1;
GAM1(23,v_lambda_p)    = rhop;
GAM1(23,v_elmbdap)     = -thetap;

%** 24 v_z ** 
GAM0(24,v_z) = 1;
GAM1(24,v_z) = rhoz;
 PSI(24,e_z) = 1;
 
%** 25 v_lambda_w ** 
GAM0(25,v_lambda_w)    =  1;
GAM0(25,v_elmbdaw)     = -1;
GAM1(25,v_lambda_w)    = rhow;
GAM1(25,v_elmbdaw)     = -thetaw;

 
%** 26 v_v ** 
GAM0(26,v_v) = 1;
GAM1(26,v_v) = rhou;
 PSI(26,e_v) = 1;
  
%** 27 v_mu ** investment shock
GAM0(27,v_mu) = 1;
GAM1(27,v_mu) = rhomu;
 PSI(27,e_mu) = 1;
 
%** 28 v_b ** intertemporal preferences hock
GAM0(28,v_b) = 1;
GAM1(28,v_b) = rhob;
 PSI(28,e_b) = 1;

%** 29 v_? ** intertemporal preferences hock
GAM0(29,v_g) = 1;
GAM1(29,v_g) = rhog;
 PSI(29,e_g) = 1;

%% 
% %**********************************************************
% %**      Expectation error
% %**********************************************************
%** 30 E[rk] **
GAM0(30,v_rk)  = 1;
GAM1(30,v_Erk) = 1;
 PPI(30,n_rk)  = 1;
 
%** 31 E[pi] **
GAM0(31,v_pi)  = 1;
GAM1(31,v_Epi) = 1;
 PPI(31,n_pi)  = 1;
 
%** 32 E[c] **
GAM0(32,v_c)  = 1;
GAM1(32,v_Ec) = 1;
 PPI(32,n_c)  = 1;

%** 33 E[lambda] **
GAM0(33,v_lambda)  = 1;
GAM1(33,v_Elambda) = 1;
 PPI(33,n_lambda)  = 1;
 
%** 34 E[i] **
GAM0(34,v_i)  = 1;
GAM1(34,v_i) = 0;
GAM1(34,v_Ei) = 1;
 PPI(34,n_i)  = 1;

%** 35 E[w] **
GAM0(35,v_w)  = 1;
GAM1(35,v_w) = 0;
GAM1(35,v_Ew) = 1;
 PPI(35,n_w)  = 1;
 
% equation 36
GAM0(36,v_elmbdap)     =  1;
PSI(36,e_lambda_p)     =  1;

% equation 37
GAM0(37,v_elmbdaw)     =  1;
PSI(37,e_lambda_w)     =  1;
%% 
%********************************************************************
%**      QZ(generalized Schur) decomposition by GENSYS
%********************************************************************
[T1,TC,T0,TY,M,TZ,TETA,GEV,RC]=gensys_mod(GAM0, GAM1, C, PSI, PPI, 1);

