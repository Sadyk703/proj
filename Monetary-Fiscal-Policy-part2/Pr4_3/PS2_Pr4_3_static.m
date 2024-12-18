function [residual, g1, g2, g3] = PS2_Pr4_3_static(y, x, params)
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

residual = zeros( 9, 1);

%
% Model equations
%

lhs =y(1);
rhs =y(1)*params(1)+params(3)*y(4)+y(8);
residual(1)= lhs-rhs;
lhs =y(3);
rhs =y(3)*0.8+y(1)*0.3+0.08*y(2)+y(9);
residual(2)= lhs-rhs;
lhs =y(4);
rhs =y(2)*params(5)-params(6)*y(5);
residual(3)= lhs-rhs;
lhs =y(5);
rhs =y(2)*(1-params(1))+params(1)*y(5)-(y(3)-y(1));
residual(4)= lhs-rhs;
lhs =y(6);
rhs =y(6)-(y(3)-y(1))/params(4);
residual(5)= lhs-rhs;
lhs =y(2);
rhs =0.8*y(6)+0.2*y(7);
residual(6)= lhs-rhs;
lhs =y(8);
rhs =y(8)*params(9)+x(1);
residual(7)= lhs-rhs;
lhs =y(9);
rhs =y(9)*params(9)+x(2);
residual(8)= lhs-rhs;
lhs =y(7);
rhs =y(7)*params(9)+x(3);
residual(9)= lhs-rhs;
if ~isreal(residual)
  residual = real(residual)+imag(residual).^2;
end
if nargout >= 2,
  g1 = zeros(9, 9);

  %
  % Jacobian matrix
  %

  g1(1,1)=1-params(1);
  g1(1,4)=(-params(3));
  g1(1,8)=(-1);
  g1(2,1)=(-0.3);
  g1(2,2)=(-0.08);
  g1(2,3)=0.2;
  g1(2,9)=(-1);
  g1(3,2)=(-params(5));
  g1(3,4)=1;
  g1(3,5)=params(6);
  g1(4,1)=(-1);
  g1(4,2)=(-(1-params(1)));
  g1(4,3)=1;
  g1(4,5)=1-params(1);
  g1(5,1)=(-1)/params(4);
  g1(5,3)=1/params(4);
  g1(6,2)=1;
  g1(6,6)=(-0.8);
  g1(6,7)=(-0.2);
  g1(7,8)=1-params(9);
  g1(8,9)=1-params(9);
  g1(9,7)=1-params(9);
  if ~isreal(g1)
    g1 = real(g1)+2*imag(g1);
  end
if nargout >= 3,
  %
  % Hessian matrix
  %

  g2 = sparse([],[],[],9,81);
if nargout >= 4,
  %
  % Third order derivatives
  %

  g3 = sparse([],[],[],9,729);
end
end
end
end
