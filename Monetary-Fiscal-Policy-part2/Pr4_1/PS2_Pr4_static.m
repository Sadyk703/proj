function [residual, g1, g2, g3] = PS2_Pr4_static(y, x, params)
%
% Status : Computes static model for Dynare
%
% Inputs : 
%   y         [M_.endo_nbr by 1] double    vector of endogenous variables in declaration order
%   x         [M_.exo_nbr by 1] double     vector of exogenous variables in declaration order
%   params    [M_.param_nbr by 1] double   vector of parameter values in declaration order
%
% Outputs:
%   residual  [M_.endo_nbr by 1] double    vector of residuals of the static model equations 
%                                          in order of declaration of the equations.
%                                          Dynare may prepend or append auxiliary equations, see M_.aux_vars
%   g1        [M_.endo_nbr by M_.endo_nbr] double    Jacobian matrix of the static model equations;
%                                                       columns: variables in declaration order
%                                                       rows: equations in order of declaration
%   g2        [M_.endo_nbr by (M_.endo_nbr)^2] double   Hessian matrix of the static model equations;
%                                                       columns: variables in declaration order
%                                                       rows: equations in order of declaration
%   g3        [M_.endo_nbr by (M_.endo_nbr)^3] double   Third derivatives matrix of the static model equations;
%                                                       columns: variables in declaration order
%                                                       rows: equations in order of declaration
%
%
% Warning : this file is generated automatically by Dynare
%           from model file (.mod)

residual = zeros( 7, 1);

%
% Model equations
%

lhs =y(1);
rhs =y(1)*params(1)+params(3)*y(4)+y(6);
residual(1)= lhs-rhs;
lhs =y(2);
rhs =y(2)-1/params(4)*(y(3)-y(1));
residual(2)= lhs-rhs;
lhs =y(3);
rhs =y(1)*params(7)+y(2)*params(8)+y(7);
residual(3)= lhs-rhs;
lhs =y(4);
rhs =y(2)*params(5)-params(6)*y(5);
residual(4)= lhs-rhs;
lhs =y(5);
rhs =y(2)*(1-params(1))+params(1)*y(5)-(y(3)-y(1));
residual(5)= lhs-rhs;
lhs =y(6);
rhs =y(6)*params(9)+x(1);
residual(6)= lhs-rhs;
lhs =y(7);
rhs =y(7)*params(9)+x(2);
residual(7)= lhs-rhs;
if ~isreal(residual)
  residual = real(residual)+imag(residual).^2;
end
if nargout >= 2,
  g1 = zeros(7, 7);

  %
  % Jacobian matrix
  %

  g1(1,1)=1-params(1);
  g1(1,4)=(-params(3));
  g1(1,6)=(-1);
  g1(2,1)=(-(1/params(4)));
  g1(2,3)=1/params(4);
  g1(3,1)=(-params(7));
  g1(3,2)=(-params(8));
  g1(3,3)=1;
  g1(3,7)=(-1);
  g1(4,2)=(-params(5));
  g1(4,4)=1;
  g1(4,5)=params(6);
  g1(5,1)=(-1);
  g1(5,2)=(-(1-params(1)));
  g1(5,3)=1;
  g1(5,5)=1-params(1);
  g1(6,6)=1-params(9);
  g1(7,7)=1-params(9);
  if ~isreal(g1)
    g1 = real(g1)+2*imag(g1);
  end
if nargout >= 3,
  %
  % Hessian matrix
  %

  g2 = sparse([],[],[],7,49);
if nargout >= 4,
  %
  % Third order derivatives
  %

  g3 = sparse([],[],[],7,343);
end
end
end
end
