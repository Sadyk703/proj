%Make sure the solution code and benchmark values mat-files are on the path 

load maindata %benchmark parameter

n=100; %number of points to evaluate the integral
w=2*pi*(-(n/2-1):1:n/2)'/n; %form vector of Fourier frequencies

%% Model solution part, this will change according to the new model        
[TT,TC,TEPS,TETA,RC] = lubiksolv(prpr,1); %this msel changes depending on problem at hand

neq=size(TT,2); %number of equations

if RC==[1;1] %determinacy
     retcode = 0;
   RR = [TEPS, zeros(neq,1)];
 
elseif RC==[1;0] %indeterminacy
     retcode = 1;
     TETA=rref(TETA').'; %reduced column echelon form for TETA
  RR = [TEPS, TETA];

end
%Check existence and determinacy, but presumably for the benchmark
%parameter everything should be fine
%% Selection matrix A - this will also change for the new model and depend
% on how variables are ordered
matA = zeros(7,37);
matA(1,18) = 1;
matA(2,9) = 1;
matA(3,15) = 1;
matA(4,3) = 1;
matA(5,5) = 1;
matA(6,7) = 1;
matA(7,10) = 1;

QQ = createcov_ls(prpr); %create covariance matrix - will differ by model


%% This is the core spectrum computation - it is universal so long as
%the corresponding inputs' names and contents from the solution conform 

ny=7; %no of observables

sqi=-1i;

id1=eye(neq);
cc=2*pi;


true_spectrum=zeros(ny*length(w),ny); %blank for spectral density over frequencies given in w
       

        for i=1:length(w)
            exe=exp(sqi*w(i));
            mat1=(id1-TT*exe)\id1; %inv(1-T1L)
            mat2=mat1';  %note that ' gives the conjugate transpose
            true_spectrum(((i-1)*ny+1):i*ny,:)=matA*mat1*RR*QQ*RR'*mat2*matA'/cc; %spectral density formula as in (3)
        end

save('true_spectrum0lsd','true_spectrum') %check result conforms with true_spectrum0asd.mat