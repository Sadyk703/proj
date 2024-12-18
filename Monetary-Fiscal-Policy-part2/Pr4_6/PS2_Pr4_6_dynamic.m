function [residual, g1, g2, g3] = PS2_Pr4_6_dynamic(y, x, params, steady_state, it_)
%
% Status : Computes dynamic model for Dynare
%
% Inputs :
%   y         [#dynamic variables by 1] double    vector of endogenous variables in the order stored
%                                                 in M_.lead_lag_incidence; see the Manual
%   x         [nperiods by M_.exo_nbr] double     matrix of exogenous variables (in declaration order)
%                                                 for all simulation periods
%   steady_state  [M_.endo_nbr by 1] double       vector of steady state values
%   params    [M_.param_nbr by 1] double          vector of parameter values in declaration order
%   it_       scalar double                       time period for exogenous variables for which to evaluate the model
%
% Outputs:
%   residual  [M_.endo_nbr by 1] double    vector of residuals of the dynamic model equations in order of 
%                                          declaration of the equations.
%                                          Dynare may prepend auxiliary equations, see M_.aux_vars
%   g1        [M_.endo_nbr by #dynamic variables] double    Jacobian matrix of the dynamic model equations;
%                                                           rows: equations in order of declaration
%                                                           columns: variables in order stored in M_.lead_lag_incidence followed by the ones in M_.exo_names
%   g2        [M_.endo_nbr by (#dynamic variables)^2] double   Hessian matrix of the dynamic model equations;
%                                                              rows: equations in order of declaration
%                                                              columns: variables in order stored in M_.lead_lag_incidence followed by the ones in M_.exo_names
%   g3        [M_.endo_nbr by (#dynamic variables)^3] double   Third order derivative matrix of the dynamic model equations;
%                                                              rows: equations in order of declaration
%                                                              columns: variables in order stored in M_.lead_lag_incidence followed by the ones in M_.exo_names
%
%
% Warning : this file is generated automatically by Dynare
%           from model file (.mod)

%
% Model equations
%

residual = zeros(11, 1);
lhs =y(5);
rhs =params(1)*y(16)+params(3)*y(8)+y(14);
residual(1)= lhs-rhs;
lhs =y(7);
rhs =y(5)*params(7)+params(8)*y(6)+y(15);
residual(2)= lhs-rhs;
lhs =y(8);
rhs =y(6)*params(5)-params(6)*y(9);
residual(3)= lhs-rhs;
lhs =y(9);
rhs =(1-params(1))*y(17)+params(1)*y(18)-(y(7)-y(16));
residual(4)= lhs-rhs;
lhs =y(10);
rhs =y(19)-(y(7)-y(16))/params(4);
residual(5)= lhs-rhs;
lhs =y(6);
rhs =y(10)*0.8+0.2*y(11);
residual(6)= lhs-rhs;
lhs =0.2*y(11)+3.6*(y(2)-y(5));
rhs =0.4*y(13)+params(1)*3.6*(y(12)-y(7));
residual(7)= lhs-rhs;
lhs =0.4*y(13);
rhs =y(12)*3.6*params(10)+y(11)*0.2*params(11);
residual(8)= lhs-rhs;
lhs =y(14);
rhs =params(9)*y(3)+x(it_, 1);
residual(9)= lhs-rhs;
lhs =y(15);
rhs =params(9)*y(4)+x(it_, 2);
residual(10)= lhs-rhs;
lhs =y(11);
rhs =params(9)*y(1)+x(it_, 3);
residual(11)= lhs-rhs;
if nargout >= 2,
  g1 = zeros(11, 22);

  %
  % Jacobian matrix
  %

  g1(1,5)=1;
  g1(1,16)=(-params(1));
  g1(1,8)=(-params(3));
  g1(1,14)=(-1);
  g1(2,5)=(-params(7));
  g1(2,6)=(-params(8));
  g1(2,7)=1;
  g1(2,15)=(-1);
  g1(3,6)=(-params(5));
  g1(3,8)=1;
  g1(3,9)=params(6);
  g1(4,16)=(-1);
  g1(4,17)=(-(1-params(1)));
  g1(4,7)=1;
  g1(4,9)=1;
  g1(4,18)=(-params(1));
  g1(5,16)=(-1)/params(4);
  g1(5,7)=1/params(4);
  g1(5,10)=1;
  g1(5,19)=(-1);
  g1(6,6)=1;
  g1(6,10)=(-0.8);
  g1(6,11)=(-0.2);
  g1(7,5)=(-3.6);
  g1(7,7)=params(1)*3.6;
  g1(7,11)=0.2;
  g1(7,2)=3.6;
  g1(7,12)=(-(params(1)*3.6));
  g1(7,13)=(-0.4);
  g1(8,11)=(-(0.2*params(11)));
  g1(8,12)=(-(3.6*params(10)));
  g1(8,13)=0.4;
  g1(9,3)=(-params(9));
  g1(9,14)=1;
  g1(9,20)=(-1);
  g1(10,4)=(-params(9));
  g1(10,15)=1;
  g1(10,21)=(-1);
  g1(11,1)=(-params(9));
  g1(11,11)=1;
  g1(11,22)=(-1);

if nargout >= 3,
  %
  % Hessian matrix
  %

  g2 = sparse([],[],[],11,484);
if nargout >= 4,
  %
  % Third order derivatives
  %

  g3 = sparse([],[],[],11,10648);
end
end
end
end
