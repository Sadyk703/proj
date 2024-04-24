%%% This file reproduces the results in the first column of Tables S11 and S12:
%%% i) the parameter vector minimizing the KL criterion between the
%%% default specification of the Carlstrom et al. (2014) model under
%%% determinacy and the parameters in the space outside the excluded 
%%% neighborhood with c=0.1.
%%% ii) the value of the minimized KL criterion and the empirical distances 
%%% for samples of size 80, 150, 200 and 1000.

% The required data files and subroutines are located in: 
%\LS_Identification\
%(lubiksolv.m, klls.m,pfhls.m,qls.m,vfhls.m, theta0lsd.mat,true_spectrum0lsd.mat)
%Additional files for  business business cycle frequencies only:
%(kllsbc.m,pfhlsbc.m,qlsbc.m,vfhlsbc.m,true_spectrum0lsdbc.mat)
%\LS_Identification\Constraints\
%(constraintlsd.m)
%\LS_Identification\Objectives\
%(kloptls3.m)
%\General\gensys_mod
%(gensys_mod.m, svdrr.m)
%\General\GA_optim.m, PSO_optim.m, psout.m

%The total runtime for GA+Multistart algorithm is about 7.5 minutes, and about 7 minutes for PSO+Multistart  
%algorithm on a Xeon E5-2665 8-core 2.4Ghz processor, using Matlab R2015a.
%%
clear
clc
addpath(genpath(pwd));
%% Algorithm selection and Matlab versions
%Select one procedure:
runga=1; %1 if using Genetic algorithm + Multistart combination, 0 otherwise
runpso=0; %1 if using Particle Swarm + Multistart combination, 0 otherwise
          %-----Note PSO is available only in R2014b and later----
 
%Specify the constraint handling for the Genetic algorithm.
NonlinCon='penalty'; %this is the preferred option, available in the matlab version 2014b and later; for earlier versions, set to 'auglag'.
%% Set up parallel computation

 numcore=2; %specify the number of cores
% 
% %for versions before R2014a
% matlabpool open local numcore
% 
% %for versions R2014a and above
ppool=parpool('local',numcore); %create parallel pool object
%% Load parameter values and bounds

load maindata %load default parameter vector (theta0)

% Set lower and upper bounds
%Parameter order: 
% [alpha	iotap	iotaw	gammaz	gammau	h	lambdapss	lambdawss	Lss	beta	Psi	xip	xiw	theta	Sadj	phipi	phidx	
%rhoR	rhoz	rhog	rhou	rhop	rhow	rhob	thetap	thetaw	rhosigma	rhonw	rhomu	hik	sigmaz	sigmag	sigmau	
%sigmarho	sigmaw	sigmab	sigmasigma	sigmanw	sigmamu	sigmamp]

%
lb=[0.1584 0.1485 0.1683 0.004851 0.005049 0.8712 0.1782 0.0792 1.37705844717914 0.988615937687238 3.3561 0.7326 0.6534 5.1183 2.4057 2.0889 0.198 0.8514 0.3663 0.98 0.3465 0.9405 0.9504 0.5742 0.7029 0.9801 0.8613 0.7722 0.8811 2.2176 0.8712 0.3564 0.5742 0.2376 0.3465 0.0297 0.0792 1.4157 3.6333 0.99];
ub=[0.1616 0.1515 0.1717 0.004949 0.005151 0.8888 0.1818 0.0808 1.40487780974842 1.00858797683243 3.4239 0.7474 0.6666 5.2217 2.4543 2.1311 0.202 0.8686 0.3737 1 0.3535 0.9595 0.9696 0.5858 0.7171 0.9999 0.8787 0.7878 0.8989 2.2624 0.8888 0.3636 0.5858 0.2424 0.3535 0.0303 0.0808 1.4443 3.7067 1.01];

%% Check the uniqueness of the solution

[C,F] = BP(prpr);
if  max(F)<1 || max(C)<1
    disp('Eigenvalues of C and F are inside the unit circle. Thus, there is a unique stable solution.')
else
    disp('Eigenvalues of C and F are outside the unit circle. Thus, there are mutliple solutions.')
end


%% Select frequencies 
bc=0; %0 for full spectrum; 1 for business cycle frequencies only

%% Specify constraints
%ns=0.1; %size of excluded neighborhood; only needed for constrained case.
ns=1; %size of excluded neighborhood; only needed for constrained case.

nrm=Inf; %set norm for the constraint function (1,2, or Inf)

numpar=length(lb);%number of parameters in the objective function

indp=[1:numpar]; %vector of parameter indices to be constrained
wgt=ones(1,length(indp)); %vector of weights for the constraint function

%% Detailed specifications; you should not have to modify them

if bc==0
    n=100; %number of points to evaluate the integral
    w=2*pi*(-(n/2-1):1:n/2)'/n; %form vector of Fourier frequencies
    resfilename=['tables_S1112_p1_v7'];
elseif bc==1
    n=500; %number of points to evaluate the integral
    w=2*pi*(-(n/2-1):1:n/2)'/n; %form vector of Fourier frequencies
    resfilename=['tables_S1112_p1_bc_v7'];
end

con1=[0,ns]; %constraint handling: con(1)=0 if algorithm handles constraint,
%con(1)=1 if penalty to be added to the objective function for infeasible
%pioints. con(2) passes the excluded neighborhood size to the constraint
%function.

con2=[1,ns]; %set constraint handling for Particle Swarm optimization.


ObjectiveFunction = @(theta0)kloptls3(theta0,con1,w,wgt,nrm,indp,bc); %set objective function for GA/multistart
ObjectiveFunctionP = @(theta0)kloptls3(theta0,con2,w,wgt,nrm,indp,bc); %set penalized objective function for PSO
ConstraintFunction=@(xest)constraintlsd(xest,ns,wgt,nrm,indp); %set constraint for the problem

dispalg='iter'; %set whether algorithm iterations are displayed.
dispint=20; %interval between displayed iterations (for PSO)

%% GA algorithm settings
if runga==1
gen=1000; %max number of generation for GA
stgenlim=50; %max number of stall generations (i.e., no improvement found)

initpop=[]; %set initial population (if smaller than popsize, MATLAB will
%randomly draw the rest. If [], the whole population is randomly drawn.
%Can be a row vector of dimension numpar or a matrix. Each row of the
%matrix is then a candidate initial value.

popsize=100; %population size
elcnt=3; %elite count - number of elite individuals retained in population

tolfunga=1e-10; %tolerance level for improvement in the objective for GA
tolconga=1e-10; %tolerance level for constraint for GA

usepga=['Always']; %Set to 'Always' to use parallel computation, otherwise to 'Never' or []
%In later versions of Matlab, 1 and 0 can also be used respectively. 
end

%% PSO algorithm settings
if runpso==1
    
swarmsize=300; %swarm size (similar concept to population size for GA)
%maxitpso=1000; %maximum number of iterations (similar concept to generations for GA)
maxitpso=1500; %maximum number of iterations (similar concept to generations for GA)
stiterlim=100; %max number of stall PSO iterations (i.e., no improvement found)
initswarm=[]; %set initial population (if smaller than swarmsize, MATLAB will
%randomly draw the rest. If [], the whole population is randomly drawn.
%Can be a row vector of dimension numpar or a matrix. Each row of the
%matrix is then a candidate initial value.

minfn=0.1; %smallest fraction of neighbors for PSO (smallest size of the 
%adaptive neighborhood)

tolfunpso=1e-06; %tolerance level for improvement in the objective for PSO

psoname=['psolsd_c',num2str(ns*100)]; %set name for a temp output file that stores the swarms (problem-based)
OutFun=@(optimValues,state)psout(optimValues,state,psoname); %output 
%function for extracting swarms from PSO runs for further local optimization.

useppso=['Always']; %Set to 'Always' to use parallel computation, otherwise to 'Never' or []


end
%% Multistart algorithm settings

numrpoints=50; %number of random starting points for Multistart

usepms=['Always']; %Set to 'Always' to use parallel computation, otherwise to 'Never' or []

% settings for fmincon
maxit=1000; % set max number of iterations it was 1000
maxfev=20000; % set max number of function evaluations it was 10 000
tolfunfmc=1e-10; %tolerance level for improvement in the objective for fmincon
tolconfmc=1e-10; %tolerance level for constraint for fmincon
tolx=1e-10; %tolerance on solution value

localg='active-set'; %set main local algorithm to be used for multistart

%% Run optimization

if runga==1
    timega=tic;
    GA_optim %run GA+Multistart
    timelga=toc(timega); %time taken by GA/Multistart
    save(resfilename) %save intermediate results
end
save(resfilename)
if runpso==1
    timepso=tic;
    PSO_optim %run PSO+Multistart
    timelpso=toc(timepso); %time taken by PSO/Multistart
    save(resfilename) %save intermediate results
end

%% Arrange results
values=[]; %blank for storing best function values
solvecs=[]; %blank for stroing solution vectors
if runga==1
    values=[values;fvalga;fvalga2];
    solvecs=[solvecs;xestga;xestga2];
end
if runpso==1
    values=[values;fvalpso;fvalpso2];
    solvecs=[solvecs;xestpso;xestpso2];
end

err=find(values<0);
values(err)=1e07; %penalize negative values that may occur due to algorithm error
indm=find(values==min(values)); %minimum value(s)
indm=indm(1); 
temp1=['thetaind=solvecs(',num2str(indm),',:)'';']; 
temp2=['kl=values(',num2str(indm),')/10000;']; 
eval(temp1); %save final parameter result
eval(temp2); %save minimized KL distance
%% Empirical distance computation
if bc==0
    ed=[pfhls(prpr,thetaind,1,1,0.05,80);pfhls(prpr,thetaind,1,1,0.05,150);pfhls(prpr,thetaind,1,1,0.05,200);pfhls(prpr,thetaind,1,1,0.05,1000)];
elseif bc==1
    ed=[pfhlsbc(prpr,thetaind,1,1,0.05,80);pfhlsbc(prpr,thetaind,1,1,0.05,150);pfhlsbc(prpr,thetaind,1,1,0.05,200);pfhlsbc(prpr,thetaind,1,1,0.05,1000)];
end
% %% Print and save results
% par=['tau     '; 'beta    '; 'kappa   '; 'psi1    '; 'psi2    '; 'rhoR    '; 'rhog    '; 'rhoz    '; 'sigR    '; 'sigg    '; 'sigz    ';'rhogz   '];
% t0=num2str(theta,3);
% ti=num2str(thetaind,3);
% t3=['KL    ';'T=80  ';'T=150 ';'T=200 ';'T=1000'];
% 
% for i=1:12
%     sp(i,:)='  ';
% end
% disp 'Table S11. Parameter values minimizing the KL criterion, Determinacy, LS (2004) model'
% disp '        (a) All parameters can vary'
% disp '           theta0     c=0.1'
% disp([par,sp,t0,sp,ti])
% 
% disp 'Table S12. KL and empirical distances between theta_c and theta_0, Determinacy, LS (2004) model'
% disp '        (a) All parameters can vary'
% disp '           c=0.1'
% disp([t3,sp(1:5,:),num2str([kl;ed],3)])
% save(resfilename)

%%
%par=['beta    '; 'kappa   '; 'rhor    '; 'psiy    '; 'psipi   '; 'sigmaz2 '; 'sigmav2 '; 'sigmapi2'];
%par=['alpha    '; 'delta   '; 'iotap    '; 'iotaw    '; 'h   '; 'Fbeta '; 'niu '; 'xip'];
%par=['alpha';	'delta';	'iotap';	'iotaw';	'h';	'Fbeta';	'niu';	'xip'];%;	'xiw';	'chi';	'Sadj';	'fp';	'fy';	'fdy';	'rhoR';	'rhoz';	'rhog';	'rhomiu';	'rholambdap';	'rholambdaw';	'rhob';	'rhomp';	'rhoARMAlambdap';	'rhoARMAlambdaw';	'gammamiu100';	'rhoupsilon';	'cnu';	'cnustar';	'cgamma';	'crp';	'crpstar';	'ctheta';	'cchi';	'rhoefp';	'rhonw';	'cb'];
%%
t0=num2str(prpr,3);
ti=num2str(thetaind,3);
t3=['KL    ';'T=80  ';'T=150 ';'T=200 ';'T=1000'];
sp = [];
for i=1:length(prpr)
    sp(i,:)='  ';
end
disp 'Table 2. Parameter values minimizing the KL criterion'
disp '        (a) All parameters can vary'
disp '           theta0     c=0.1'
disp([sp,t0,sp,ti])

disp 'Table 3. KL and empirical distances between theta_c and theta_0'
disp '        (a) All parameters can vary'
disp '           c=0.1'
disp([t3,sp(1:5,:),num2str([kl;ed],3)])
save(resfilename)
%% Close parallel pool

% %for versions before R2014a
% matlabpool close
% 
% %for versions R2014a and above
delete(ppool) %delete the parallel pool object