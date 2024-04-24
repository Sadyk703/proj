function P = pfhls(theta1,theta2,msel1,msel2,a,n)

%Function to compute the empirical distance measure (pfh) for significance
%level a, for sample size n, Lubik and Schorfheide (2004) model

%Inputs:
% theta1 - parameter vector of the null model
% msel1 - model selector for the null model (=1 for standard LS(2004);
%=2 for Taylor rule with expected inflation,=3 for output growth Taylor rule)

% theta2 - parameter vector of the alternative model
% msel2 - model selector for the alternative model (=1 for standard LS(2004);
%=2 for Taylor rule with expected inflation;=3 for output growth Taylor rule)

% a - significance level. E.g., a=0.05 for 5% level.

% n - sample size

z=norminv(1-a,0,1); %Normal critical value
q=qls(theta1,theta2,msel1,msel2,a,n); %q-alpha

temp=(q-sqrt(n)*klls(theta2,theta1,msel2,msel1))/(sqrt(vfhls(theta2,theta1,msel2,msel1))); %term inside brackets

P=1-normcdf(temp); %final probability

end