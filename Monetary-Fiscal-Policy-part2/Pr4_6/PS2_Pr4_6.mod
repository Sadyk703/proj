%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Problem Set 2, Problem 4%%%%%%%%%%%%%%%%%%%%%%%%
% Performed: Sodik Umurzakov
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Problem 4.6%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Endo variables
var pi y r x s c g b t eps_pi eps_r;

%Exo variables
varexo eps_pi_shock eps_r_shock eps_g_shock;

% Set parameter value
parameters beta lambda kappa sigma eta nu phi_pi phi_y rho phi_b phi_g;

% Set initial values of parameters
beta = 0.99;
lambda = 0.66;
kappa = ((1-lambda)*(1-beta*lambda))/lambda;
sigma = 1;
eta = 1.2;
nu = 0; %according to the conditions in the PS2_problem4.2 ( e.g. without financial frictions)
phi_pi = 1.5;
phi_y = 0.125;
phi_b = 0.043;
phi_g = 0.124;
rho = 0.9;

model;
    %Inflation
    pi = beta*pi(+1) + kappa*x + eps_pi;
    
    %Interest rate (original)
    r = phi_pi*pi + phi_y*y + eps_r;
    
    %Marginal cost
    x = eta*y - nu*s;

    %Stock price
    s = (1-beta)*y(+1) + beta*s(+1) - (r - pi(+1));
    
    %Consumption
    c = c(+1) - (r - pi(+1))/sigma;
    
    %Output
    y = 0.8*c+0.2*g;
    
    %Government BC
    0.2*g + 3.6*(b(-1) - pi) = 0.4*t + beta*3.6*(b - r);

    %Lump-sum tax
    0.4*t = phi_b*3.6*b + phi_g*0.2*g;

    %AR(1) process of inflation
    eps_pi = rho*eps_pi(-1) + eps_pi_shock;
    
    %AR(1) process of interest rate
    eps_r = rho*eps_r(-1) + eps_r_shock;
    
    %AR(1) process of government spending
    g = rho *g(-1) + eps_g_shock;
end;

%Set values in steady state
initval;
    pi = 0.0;
    y = 0.0;
    r = 0.0;
    x = 0.0;
    s = 0.0;
    c = 0.0;
    g = 0.0;
    b = 0.0;
    t = 0.0;
    eps_pi = 0.0;
    eps_r = 0.0;
end;

shocks;
    var eps_pi_shock = 0;
    var eps_r_shock = 0;
    var eps_g_shock = 1;
end;

stoch_simul(order=1, irf=40);

% Save IRF data to a MAT file
irfs = oo_.irfs; % Extract the IRF data
save('IRFs_data.mat', 'irfs');




