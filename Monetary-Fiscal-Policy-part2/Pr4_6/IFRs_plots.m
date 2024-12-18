%%




%%
% Load IRF data
load('IRFs_data.mat');

% Time horizon for IRFs
horizon = 1:40;

% Variable names and corresponding titles
variables = {'pi_eps_g_shock', 'y_eps_g_shock', 'r_eps_g_shock', 'x_eps_g_shock', ...
             's_eps_g_shock', 'c_eps_g_shock', 'g_eps_g_shock', 'b_eps_g_shock', ...
             't_eps_g_shock'};
titles = {'Inflation (\pi)', 'Output (y)', 'Interest Rate (r)', 'Marginal Cost (x)', ...
          'Stock Price (s)', 'Consumption (c)', 'Government Spending (g)', ...
          'Debt (b)', 'Taxes (t)'};

% Number of variables
num_vars = length(variables);

% Create a figure
figure;

% Dynamically create subplots for each variable
for i = 1:num_vars
    subplot(ceil(num_vars / 3), 3, i); % Adjust layout dynamically
    plot(horizon, oo_.irfs.(variables{i}), 'LineWidth', 2);
    title(titles{i}, 'FontSize', 14);
    xlabel('Periods', 'FontSize', 12);
    ylabel('Deviation', 'FontSize', 12);
    grid on;
end


% Adjust tittle
sgtitle('IRFs to fiscal policy shock', 'FontSize', 16);
%Save
saveas(gcf, 'IRFs_to_fiscal_policy.png');



%%




