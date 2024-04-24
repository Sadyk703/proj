function q = qls(theta1,theta2,msel1,msel2,a,n)

%function to compute q-alpha for sig. level a, sample size n, LS(2004)
%model

z=norminv(1-a,0,1); %Normal critical value
q=(-sqrt(n)*klls(theta1,theta2,msel1,msel2)) + sqrt(vfhls(theta1,theta2,msel1,msel2))*z; %q-alpha


end