function [c, ceq] = constraintlsd(xest,ns,wgt,nrm,indp)
%Constraint function for global identification under determinacy,LS(2004)
%model.
%Keeps the specified norm between two parameter vectors >=ns (neighborhood size)

%Inputs:
%xest - 1x(p+q) candidate vector of parameters
%ns - neighborhood size (denoted "c" in the paper)
%wgt - 1x(p+q-1) user-specified vector of weights
%nrm - specified norm, takes inputs 1 (for L-1 norm), 2(for L-2 norm) and Inf (for infinity norm)
%indp - vector of indices for parameters to be constrained, in increasing
%order. 

load maindata;

xcon=xest(indp); %select the constrained parameters

   c = ns-norm((xcon-prpr(indp)')./wgt,nrm);
   ceq = [];
end