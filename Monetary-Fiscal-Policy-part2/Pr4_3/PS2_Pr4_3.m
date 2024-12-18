%
% Status : main Dynare file
%
% Warning : this file is generated automatically by Dynare
%           from model file (.mod)

if isoctave || matlab_ver_less_than('8.6')
    clear all
else
    clearvars -global
    clear_persistent_variables(fileparts(which('dynare')), false)
end
tic0 = tic;
% Save empty dates and dseries objects in memory.
dates('initialize');
dseries('initialize');
% Define global variables.
global M_ options_ oo_ estim_params_ bayestopt_ dataset_ dataset_info estimation_info ys0_ ex0_
options_ = [];
M_.fname = 'PS2_Pr4_3';
M_.dynare_version = '4.5.7';
oo_.dynare_version = '4.5.7';
options_.dynare_version = '4.5.7';
%
% Some global variables initialization
%
global_initialization;
diary off;
diary('PS2_Pr4_3.log');
M_.exo_names = 'eps_pi_shock';
M_.exo_names_tex = 'eps\_pi\_shock';
M_.exo_names_long = 'eps_pi_shock';
M_.exo_names = char(M_.exo_names, 'eps_r_shock');
M_.exo_names_tex = char(M_.exo_names_tex, 'eps\_r\_shock');
M_.exo_names_long = char(M_.exo_names_long, 'eps_r_shock');
M_.exo_names = char(M_.exo_names, 'eps_g_shock');
M_.exo_names_tex = char(M_.exo_names_tex, 'eps\_g\_shock');
M_.exo_names_long = char(M_.exo_names_long, 'eps_g_shock');
M_.endo_names = 'pi';
M_.endo_names_tex = 'pi';
M_.endo_names_long = 'pi';
M_.endo_names = char(M_.endo_names, 'y');
M_.endo_names_tex = char(M_.endo_names_tex, 'y');
M_.endo_names_long = char(M_.endo_names_long, 'y');
M_.endo_names = char(M_.endo_names, 'r_alter');
M_.endo_names_tex = char(M_.endo_names_tex, 'r\_alter');
M_.endo_names_long = char(M_.endo_names_long, 'r_alter');
M_.endo_names = char(M_.endo_names, 'x');
M_.endo_names_tex = char(M_.endo_names_tex, 'x');
M_.endo_names_long = char(M_.endo_names_long, 'x');
M_.endo_names = char(M_.endo_names, 's');
M_.endo_names_tex = char(M_.endo_names_tex, 's');
M_.endo_names_long = char(M_.endo_names_long, 's');
M_.endo_names = char(M_.endo_names, 'c');
M_.endo_names_tex = char(M_.endo_names_tex, 'c');
M_.endo_names_long = char(M_.endo_names_long, 'c');
M_.endo_names = char(M_.endo_names, 'g');
M_.endo_names_tex = char(M_.endo_names_tex, 'g');
M_.endo_names_long = char(M_.endo_names_long, 'g');
M_.endo_names = char(M_.endo_names, 'eps_pi');
M_.endo_names_tex = char(M_.endo_names_tex, 'eps\_pi');
M_.endo_names_long = char(M_.endo_names_long, 'eps_pi');
M_.endo_names = char(M_.endo_names, 'eps_r');
M_.endo_names_tex = char(M_.endo_names_tex, 'eps\_r');
M_.endo_names_long = char(M_.endo_names_long, 'eps_r');
M_.endo_partitions = struct();
M_.param_names = 'beta';
M_.param_names_tex = 'beta';
M_.param_names_long = 'beta';
M_.param_names = char(M_.param_names, 'lambda');
M_.param_names_tex = char(M_.param_names_tex, 'lambda');
M_.param_names_long = char(M_.param_names_long, 'lambda');
M_.param_names = char(M_.param_names, 'kappa');
M_.param_names_tex = char(M_.param_names_tex, 'kappa');
M_.param_names_long = char(M_.param_names_long, 'kappa');
M_.param_names = char(M_.param_names, 'sigma');
M_.param_names_tex = char(M_.param_names_tex, 'sigma');
M_.param_names_long = char(M_.param_names_long, 'sigma');
M_.param_names = char(M_.param_names, 'eta');
M_.param_names_tex = char(M_.param_names_tex, 'eta');
M_.param_names_long = char(M_.param_names_long, 'eta');
M_.param_names = char(M_.param_names, 'nu');
M_.param_names_tex = char(M_.param_names_tex, 'nu');
M_.param_names_long = char(M_.param_names_long, 'nu');
M_.param_names = char(M_.param_names, 'phi_pi');
M_.param_names_tex = char(M_.param_names_tex, 'phi\_pi');
M_.param_names_long = char(M_.param_names_long, 'phi_pi');
M_.param_names = char(M_.param_names, 'phi_y');
M_.param_names_tex = char(M_.param_names_tex, 'phi\_y');
M_.param_names_long = char(M_.param_names_long, 'phi_y');
M_.param_names = char(M_.param_names, 'rho');
M_.param_names_tex = char(M_.param_names_tex, 'rho');
M_.param_names_long = char(M_.param_names_long, 'rho');
M_.param_partitions = struct();
M_.exo_det_nbr = 0;
M_.exo_nbr = 3;
M_.endo_nbr = 9;
M_.param_nbr = 9;
M_.orig_endo_nbr = 9;
M_.aux_vars = [];
M_.Sigma_e = zeros(3, 3);
M_.Correlation_matrix = eye(3, 3);
M_.H = 0;
M_.Correlation_matrix_ME = 1;
M_.sigma_e_is_diagonal = 1;
M_.det_shocks = [];
options_.block=0;
options_.bytecode=0;
options_.use_dll=0;
M_.hessian_eq_zero = 1;
erase_compiled_function('PS2_Pr4_3_static');
erase_compiled_function('PS2_Pr4_3_dynamic');
M_.orig_eq_nbr = 9;
M_.eq_nbr = 9;
M_.ramsey_eq_nbr = 0;
M_.set_auxiliary_variables = exist(['./' M_.fname '_set_auxiliary_variables.m'], 'file') == 2;
M_.lead_lag_incidence = [
 0 5 14;
 0 6 15;
 1 7 0;
 0 8 0;
 0 9 16;
 0 10 17;
 2 11 0;
 3 12 0;
 4 13 0;]';
M_.nstatic = 1;
M_.nfwrd   = 4;
M_.npred   = 4;
M_.nboth   = 0;
M_.nsfwrd   = 4;
M_.nspred   = 4;
M_.ndynamic   = 8;
M_.equations_tags = {
};
M_.static_and_dynamic_models_differ = 0;
M_.exo_names_orig_ord = [1:3];
M_.maximum_lag = 1;
M_.maximum_lead = 1;
M_.maximum_endo_lag = 1;
M_.maximum_endo_lead = 1;
oo_.steady_state = zeros(9, 1);
M_.maximum_exo_lag = 0;
M_.maximum_exo_lead = 0;
oo_.exo_steady_state = zeros(3, 1);
M_.params = NaN(9, 1);
M_.NNZDerivatives = [33; -1; -1];
M_.params( 1 ) = 0.99;
beta = M_.params( 1 );
M_.params( 2 ) = 0.66;
lambda = M_.params( 2 );
M_.params( 3 ) = (1-M_.params(2))*(1-M_.params(2)*M_.params(1))/M_.params(2);
kappa = M_.params( 3 );
M_.params( 4 ) = 1;
sigma = M_.params( 4 );
M_.params( 5 ) = 1.2;
eta = M_.params( 5 );
M_.params( 6 ) = 0;
nu = M_.params( 6 );
M_.params( 7 ) = 1.5;
phi_pi = M_.params( 7 );
M_.params( 8 ) = 0.125;
phi_y = M_.params( 8 );
M_.params( 9 ) = 0.9;
rho = M_.params( 9 );
%
% INITVAL instructions
%
options_.initval_file = 0;
oo_.steady_state( 1 ) = 0.0;
oo_.steady_state( 2 ) = 0.0;
oo_.steady_state( 3 ) = 0.0;
oo_.steady_state( 4 ) = 0.0;
oo_.steady_state( 5 ) = 0.0;
oo_.steady_state( 6 ) = 0.0;
oo_.steady_state( 7 ) = 0.0;
oo_.steady_state( 8 ) = 0.0;
oo_.steady_state( 9 ) = 0.0;
if M_.exo_nbr > 0
	oo_.exo_simul = ones(M_.maximum_lag,1)*oo_.exo_steady_state';
end
if M_.exo_det_nbr > 0
	oo_.exo_det_simul = ones(M_.maximum_lag,1)*oo_.exo_det_steady_state';
end
%
% SHOCKS instructions
%
M_.exo_det_length = 0;
M_.Sigma_e(1, 1) = 0;
M_.Sigma_e(2, 2) = 0;
M_.Sigma_e(3, 3) = 1;
options_.irf = 40;
options_.order = 1;
var_list_ = char();
info = stoch_simul(var_list_);
irfs_christiano = oo_.irfs; 
save('IRFs_Christiano.mat', 'irfs_christiano');
save('PS2_Pr4_3_results.mat', 'oo_', 'M_', 'options_');
if exist('estim_params_', 'var') == 1
  save('PS2_Pr4_3_results.mat', 'estim_params_', '-append');
end
if exist('bayestopt_', 'var') == 1
  save('PS2_Pr4_3_results.mat', 'bayestopt_', '-append');
end
if exist('dataset_', 'var') == 1
  save('PS2_Pr4_3_results.mat', 'dataset_', '-append');
end
if exist('estimation_info', 'var') == 1
  save('PS2_Pr4_3_results.mat', 'estimation_info', '-append');
end
if exist('dataset_info', 'var') == 1
  save('PS2_Pr4_3_results.mat', 'dataset_info', '-append');
end
if exist('oo_recursive_', 'var') == 1
  save('PS2_Pr4_3_results.mat', 'oo_recursive_', '-append');
end


disp(['Total computing time : ' dynsec2hms(toc(tic0)) ]);
if ~isempty(lastwarn)
  disp('Note: warning(s) encountered in MATLAB/Octave code')
end
diary off
