function X = kloptls3(theta0,con,w,wgt,nrm,indp,bc)

%Function to compute KL distance between models for distance minimization
%in LS(2004) model.
%all parameters can vary, current inflation rule, determinacy.

%con=[pen,c] 
%con = [1,ns];
%pen=1 triggers the penalty in case of constraint violation 

% c denotes the size of excluded neighborhood 
%w - vector of frequencies between [-pi,pi]
%bc=1 then the function uses only the business cycle frequencies in
%computation; 0 for using the full spectrum.

%wgt,nrm,indp - these inputs are needed for evaluating the constraint
%function should penalty need to be applied. Refer to
%constraintlsi.m for details.
warning('off','all');
nw=length(w);

%evaluate constraint and apply flat penalty if violated
if con(1)==1
    conval=constraintlsd(theta0,con(2),wgt,nrm,indp);
if conval>1e-10
    X=1e10;
    return
end
end

  
%% solve the model
%Specify the monetary policy rule 

ny=7; %number of variables
[TT,TC,TEPS,TETA,RC] = lubiksolv(theta0,1); 
if isempty(TT)
    disp('empty solution: nonexistence/numerical issue') 
    X=1e10;
    return
end
neq=size(TT,2); %number of equations
dck=rcond(eye(neq)-TT);

if dck<1e-10
    X=1e10;
    return
end

if RC==[1;1] %determinacy
     retcode = 0;
   RR = [TEPS, zeros(neq,1)];
 
elseif RC==[1;0] %indeterminacy
     retcode = 1;
     TETA=rref(TETA').'; %reduced column echelon form for TETA
  RR = [TEPS, TETA];
else%no equilibrium exists/numerical problems.
    X=1e10;
    return
end

A = zeros(7,37);
A(1,18) = 1;
A(2,9) = 1;
A(3,15) = 1;
A(4,3) = 1;
A(5,5) = 1;
A(6,7) = 1;
A(7,10) = 1;

QQ = createcov_ls(theta0);

dd=min(eig(QQ));

if bc==0 %full spectrum

    load true_spectrum0lsd %benchmark model spectrum (full)

%preparations to compute spectrum
term1=0;  
term2=0;
%tempeye=eye(neq);

% [-49,-1] and [1,49]
for i=1:(nw/2-1); %compute spectrum
exe=exp(-1i*w(i));

mat1=(eye(neq)-TT*exe)\eye(neq);
mat2=mat1';  %note that ' gives the conjugate transpose
spec=A*mat1*RR*QQ*RR'*mat2*A'/(2*pi); %compute spectrum using representation in the paper

zz=spec\true_spectrum(((i-1)*ny+1):i*ny,:);

term1=term1-2*log(det(zz)); 
term2=term2+2*trace(zz);

end

%the zero frequency
i=nw/2;    
exe=exp(-1i*w(i));

mat1=(eye(neq)-TT*exe)\eye(neq);
mat2=mat1';  %note that ' gives the conjugate transpose
spec=A*mat1*RR*QQ*RR'*mat2*A'/(2*pi); %compute spectrum using representation in the paper
zz=spec\true_spectrum(((i-1)*ny+1):i*ny,:);

term1=term1-log(det(zz)); 
term2=term2+trace(zz);
%the last frequency    
i=nw;
exe=exp(-1i*w(i)); 

mat1=(eye(neq)-TT*exe)\eye(neq);
mat2=mat1';  %note that ' gives the conjugate transpose
spec=A*mat1*RR*QQ*RR'*mat2*A'/(2*pi); %compute spectrum using representation in the paper
zz=spec\true_spectrum(((i-1)*ny+1):i*ny,:);

term1=term1-log(det(zz)); 
term2=term2+trace(zz);

WL=real(term1+term2); %small complex residual of order e-15i sometimes remains

X=WL/(2*nw)-1.5; %KL


elseif bc==1 %bc frequencies only
    load true_spectrum0lsdbc %benchmark model spectrum (bc only) 
    
    %identify BC frequencies
    ind1=find(w>=-pi/3 & w<=-pi/16);
    ind2=find(w>=pi/16 & w<=pi/3);
    idbc=[ind1;ind2];

%update to BC frequencies for computation
w2=w(idbc);
n2=length(w2);
term1=0;  
term2=0;
for i=1:n2; %compute spectrum
exe=exp(-1i*w2(i));

mat1=(eye(neq)-TT*exe)\eye(neq);
mat2=mat1';  %note that ' gives the conjugate transpose

ftheta=A*mat1*RR*QQ*RR'*mat2*A'/(2*pi); %compute spectrum using representation in the paper

term1=term1-log(det(ftheta\true_spectrum(((i-1)*ny+1):i*ny,:))); %second term in the KL


term2=term2+trace(ftheta\true_spectrum(((i-1)*ny+1):i*ny,:)); %first term in the KL

end
WL=real(term1+term2); %small complex residual of order e-15i sometimes remains

X=WL/(2*nw)-1.5*(n2/nw); %final answer: KL (it was 1.5, we can use 2)
end



X=X*10000;

if dd<0
    X=1e10;
end



end