function [residual, g1, g2, g3] = PS2_Pr4_2_2_dynamic(y, x, params, steady_state, it_)
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

residual = zeros(9, 1);
lhs =y(4);
rhs =params(1)*y(13)+params(3)*y(7)+y(11);
residual(1)= lhs-rhs;
lhs =y(6);
rhs =y(4)*params(7)+params(8)*y(5)+y(12);
residual(2)= lhs-rhs;
lhs =y(7);
rhs =y(5)*params(5)-params(6)*y(8);
residual(3)= lhs-rhs;
lhs =y(8);
rhs =(1-params(1))*y(14)+params(1)*y(15)-(y(6)-y(13));
residual(4)= lhs-rhs;
lhs =y(9);
rhs =y(16)-(y(6)-y(13))/params(4);
residual(5)= lhs-rhs;
lhs =y(5);
rhs =y(9)*0.8+0.2*y(10);
residual(6)= lhs-rhs;
lhs =y(11);
rhs =params(9)*y(2)+x(it_, 1);
residual(7)= lhs-rhs;
lhs =y(12);
rhs =params(9)*y(3)+x(it_, 2);
residual(8)= lhs-rhs;
lhs =y(10);
rhs =params(9)*y(1)+x(it_, 3);
residual(9)= lhs-rhs;
if nargout >= 2,
  g1 = zeros(9, 19);

  %
  % Jacobian matrix
  %

  g1(1,4)=1;
  g1(1,13)=(-params(1));
  g1(1,7)=(-params(3));
  g1(1,11)=(-1);
  g1(2,4)=(-params(7));
  g1(2,5)=(-params(8));
  g1(2,6)=1;
  g1(2,12)=(-1);
  g1(3,5)=(-params(5));
  g1(3,7)=1;
  g1(3,8)=params(6);
  g1(4,13)=(-1);
  g1(4,14)=(-(1-params(1)));
  g1(4,6)=1;
  g1(4,8)=1;
  g1(4,15)=(-params(1));
  g1(5,13)=(-1)/params(4);
  g1(5,6)=1/params(4);
  g1(5,9)=1;
  g1(5,16)=(-1);
  g1(6,5)=1;
  g1(6,9)=(-0.8);
  g1(6,10)=(-0.2);
  g1(7,2)=(-params(9));
  g1(7,11)=1;
  g1(7,17)=(-1);
  g1(8,3)=(-params(9));
  g1(8,12)=1;
  g1(8,18)=(-1);
  g1(9,1)=(-params(9));
  g1(9,10)=1;
  g1(9,19)=(-1);

if nargout >= 3,
  %
  % Hessian matrix
  %

  g2 = sparse([],[],[],9,361);
if nargout >= 4,
  %
  % Third order derivatives
  %

  g3 = sparse([],[],[],9,6859);
end
end
end
end
