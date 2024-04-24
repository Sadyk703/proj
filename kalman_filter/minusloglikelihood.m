% This function calculates the  minus loglikelihood for a fixed
% parametrisation
function LL = minusloglikelihood(param,data)

%% Solution of the model for given parameter values in the following form:
% z_t = A*z(t-1)+B*\epsilon_t
% y_t = C*z_t

[A, ~, B, ~, ~] = Carlstrom_solve(param,1);
if isempty(A)
    LL = 1000000;
    return
end

% adjustment of B to make shocks i.i.d. with identity covariance matrix

% sigmaz     = param(31); % this variables are not used here and
% sigmag     = param(32); % provided just to explain from where do they
% sigmau     = param(33); % come from
% sigmarho   = param(34);
% sigmaw     = param(35);
% sigmab     = param(36);
% sigmasigma = param(37);
% sigmanw    = param(38);
% sigmamu    = param(39);
% sigmamp    = param(40);

% transform vector of errors eps_t into vector of standard normal with
% coefficient E: E*epsstandard_t
E = zeros(size(B,2),size(B,2)); 

for j = 1:10
    E(j,j) = param(30+j);
end

B = B*E; %

% ovservables' selection matrix
C = zeros(5,37);
C(1,10) = 1; % 1 - federal funds rate (FFR)
C(2,15) = 1; % 2 - private investment (FPI)
C(3,7)  = 1; % 3 - GDPDEF deflator-based inflation (GDPDEF)
C(4,5)  = 1; % 4 - WAGE wages (WAGE)
C(5,9)  = 1; % 5 - CONS consumption (CONS)

dim = length(A);
T = size(data,1); %length of time-series
%% Log-Likelihood calculation

ll    = 0; % initial likelihood value

% initialisation for t = 0;
z     = zeros(dim,1); 
cov_z = 10*eye(dim);

for t = 1:T
    % prediction step
    z_pred     = A*z;
    cov_z_pred = A*cov_z*A.' + B*B.';
    cov_z_pred = (cov_z_pred + cov_z_pred.')/2; % correction of floating error
    y          = C*z_pred;
    cov_y      = C*cov_z_pred*C.';              % + 0.001*eye(size(C,1)) — in case det(cov_y) becomes close to zero;
    cov_y      = (cov_y + cov_y.')/2;           % correction of floating error
    % likelihood updating
    ll         = ll + log(mvnpdf(data(t,1:5),y.',cov_y));
    
    % correction step
    z          = z_pred + cov_z_pred*C.'*inv(cov_y)*(data(t,1:5).' - y);
    % kgain      = cov_z_pred*C.'*inv(cov_y); 
    cov_z      = cov_z_pred - cov_z_pred*C.'*inv(cov_y)*C*cov_z_pred;
    % cov_z      = (eye(dim) - kgain*C)*cov_z_pred*(eye(dim) - kgain*C).'; % activate if Joseph form is preferrable
    cov_z      = (cov_z + cov_z.')/2;
 end
%disp(det(cov_y));
LL = ll*(-1); % (-1) is needed since fmincon in the main file is minimising,
              % while we need to maximise


end