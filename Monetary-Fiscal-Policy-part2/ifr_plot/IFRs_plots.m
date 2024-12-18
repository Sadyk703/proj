%% Plot IFRs for all problems

% IRF data for both cases
load('IRFs_with_frictions.mat');
load('IRFs_without_frictions.mat');

% Time horizon for IRFs
horizon = 1:40;

% Variable names and corresponding titles
variables = {'pi_eps_r_shock', 'y_eps_r_shock', 'r_eps_r_shock', 'x_eps_r_shock', 's_eps_r_shock', 'eps_r_eps_r_shock'};
titles = {'Inflation (\pi)', 'Output (y)', 'Interest Rate (r)', ...
          'Marginal Cost (x)', 'Stock Price (s)', 'Monetary Policy Shock (\epsilon_r)', };

% Number of variables
num_vars = length(variables);

% Create a figure
figure;

% Loop through variables and plot for both cases
for i = 1:num_vars
    subplot(ceil(num_vars / 2), 2, i);
    % Plot with financial frictions
    plot(horizon, irfs_with.(variables{i}), 'b-', 'LineWidth', 2, 'DisplayName', 'With Frictions');
    hold on;
    % Plot without financial frictions
    plot(horizon, irfs_without.(variables{i}), 'r--', 'LineWidth', 2, 'DisplayName', 'Without Frictions');
    title(titles{i}, 'FontSize', 14);
    xlabel('Periods', 'FontSize', 12);
    ylabel('Deviation', 'FontSize', 12);
    grid on;
    legend('show', 'FontSize', 10, 'Location', 'Best');
    hold off;
end

% Add a title
sgtitle('Comparison of IRFs: with and without financial frictions', 'FontSize', 16);

% Save the plot
saveas(gcf, 'IRFs_Comparison.png');


%%
%Problem 2.2-2.3

% IRF data for both cases
load('IRFs_Taylor.mat');
%irfs_taylor = oo_.irfs;
load('IRFs_Christiano.mat');
%irfs_christiano = oo_.irfs;

% Time horizon for IRFs
horizon = 1:40;

% Variable names and corresponding titles
variables_christiano = {'pi_eps_g_shock', 'y_eps_g_shock', 'r_alter_eps_g_shock', 'x_eps_g_shock', ...
             's_eps_g_shock', 'c_eps_g_shock', 'g_eps_g_shock'};
titles = {'Inflation (\pi)', 'Output (y)', 'Interest Rate (r)', 'Marginal Cost (x)', ...
          'Stock Price (s)', 'Consumption (c)', 'Government Spending (g)'};

variables = {'pi_eps_g_shock', 'y_eps_g_shock', 'r_eps_g_shock', 'x_eps_g_shock', ...
             's_eps_g_shock', 'c_eps_g_shock', 'g_eps_g_shock'};      
% Number of variables
num_vars = length(variables);

% Create a figure
figure;

% Loop through variables and plot for both cases
for i = 1:num_vars
    subplot(ceil(num_vars / 2), 2, i);
    % Plot for Taylor Rule
    plot(horizon, irfs_taylor.(variables{i}), 'b-', 'LineWidth', 2, 'DisplayName', 'Taylor Rule');
    hold on;
    % Plot for Christiano et al. Rule
    plot(horizon, irfs_christiano.(variables_christiano{i}), 'r--', 'LineWidth', 2, 'DisplayName', 'Christiano Rule');
    title(titles{i}, 'FontSize', 14);
    xlabel('Periods', 'FontSize', 12);
    ylabel('Deviation', 'FontSize', 12);
    grid on;
    legend('show', 'FontSize', 10, 'Location', 'Best');
    hold off;
end

% Add a global title
sgtitle('Comparison of IRFs: Taylor rule and Christiano et al. (2005) rule', 'FontSize', 16);

% Save the plot as a high-resolution image
saveas(gcf, 'IRFs_Comparison_Taylor_Christiano.png');


%%
%Problem 4.6

% IRF data
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