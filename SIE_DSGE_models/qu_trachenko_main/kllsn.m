function KL = kllsn(theta1,theta2,msel1,msel2,n)
%computes KL distance for LS(2004)-type models, the number of evaluation
%points is treated as an input

%solve both models
[TT1,TC1,TEPS1,TETA1,RC1] = lubiksolv(theta1,msel1);
[TT2,TC2,TEPS2,TETA2,RC2] = lubiksolv(theta2,msel2);

%check whether both models have solutions
if isempty(TT1) || isempty(TT2)
    disp('empty solution: nonexistence/numerical issue') 
    KL=[];
    return
end

neq1=size(TT1,2); %number of equations - model 1
neq2=size(TT2,2); %number of equations - model 2

%observables selection matrix
A=[4	0	0	0	0	0	0;...
   0	1	0	0	0	0	0;...
   0	0	4	0	0	0	0];

%the integral [-pi,pi] is approximated using n points

omi=2*pi*(-(n/2-1):1:(n/2))'/n; %form vector of Fourier frequencies for approximation of the integral 

dimy=size(A,1); %number of observables
sp_mat1=zeros(dimy*n,dimy); 
sp_mat2=sp_mat1;

%compute model 1 spectrum over all frequencies

if RC1==[1;1] %determinacy in model 1
    RR1 = [TEPS1,zeros(neq1,1)];
    QQ1=createcov_ls(theta1); %diagonal covariance matrix
    
    elseif RC1==[1;0] %if indeterminacy in model 1
    TETA1=rref(TETA1').'; %reduced column echelon form for TETA
    RR1 = [TEPS1,TETA1];
    QQ1=createcov_ls(theta1); %augmented covariance matrix

    else
    KL=[];
    disp('Nonexistence or numerical problems in model1')
    return
    
end

for i=1:n %changed
exe=exp(-1i*omi(i));
mat1=(eye(neq1)-TT1*exe)\eye(neq1); %inv(1-T1L)
mat2=mat1';  %note that ' gives the conjugate transpose


sp_mat1(((i-1)*dimy+1):(i*dimy),:)=A*mat1*RR1*QQ1*RR1'*mat2*A'/(2*pi);

end

   


clear mat1 mat2 exe

%compute model 2 spectrum over all frequencies

if RC2==[1;1] %determinacy in model 2
    RR2 = [TEPS2,zeros(neq2,1)];
    QQ2=createcov_ls(theta2); %diagonal covariance matrix
    
    elseif RC2==[1;0] %if indeterminacy in model 2
    TETA2=rref(TETA2').'; %reduced column echelon form for TETA
    RR2 = [TEPS2,TETA2];
    QQ2=createcov_ls(theta2); %augmented covariance matrix

    else
    KL=[];
    disp('Nonexistence or numerical problems in model2')
    return    
end

    
for i=1:n %changed
exe=exp(-1i*omi(i));
mat1=(eye(neq2)-TT2*exe)\eye(neq2); %inv(1-T1L)
mat2=mat1';  %note that ' gives the conjugate transpose


sp_mat2(((i-1)*dimy+1):(i*dimy),:)=A*mat1*RR2*QQ2*RR2'*mat2*A'/(2*pi);

end


%approximate the integral

term1=0;
term2=0;
for i=1:n %changed
    
   term1=term1-log(det(sp_mat2(((i-1)*dimy+1):(i*dimy),:)\sp_mat1(((i-1)*dimy+1):(i*dimy),:))); %second term in the KL


    term2=term2+trace(sp_mat2(((i-1)*dimy+1):(i*dimy),:)\sp_mat1(((i-1)*dimy+1):(i*dimy),:)); %second term in the likelihood
end



WL=real(term1+term2); %small complex residual of order e-15i sometimes remains

KL=WL/(2*n)-3/2; %final answer: KL

end