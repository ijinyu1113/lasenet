function h = sim_uniform_LVLS(nr,nc,subplots)
% File paths for saving data
sim_folder = 'simulations_uniform_LVLS';
plot_folder = 'plots_uniform_LVLS';
sum_folder = 'sum_uniform_LVLS';
fsum = fullfile(sum_folder, 'summary.mat');
mkdir(sim_folder); % Ensure folder exists
mkdir(plot_folder); % Ensure folder exists
mkdir(sum_folder);  % Ensure summary folder exists

% User-defined settings for random sampling
num_sims = 5000; % Approximately 5000 simulations

% Define parameter ranges for uniform sampling
V0_range = [0.3, 0.3];     % From V0_start to V0_end
S0_range = [0.0825, 0.0825]; % From S0_start to S0_end
lambda_range = [0.125, 0.2]; % From lambda_start to lambda_end
omega_range = [0, 1.5];   % From omega_start to omega_end
error_range = [-3, 3];    % For error1, error2, error3

base_x = readmatrix('Wind_Conditions.xlsx');
base_x = base_x(:,1);
base_x = base_x(81:end);
base_x = base_x * 10000;
N = size(base_x, 1);

sim_count = 118; % Track simulation index

% Iterate for the desired number of simulations
while sim_count <= num_sims
    fprintf('Simulation %d\n', sim_count);

    % Sample parameters from uniform distributions
    v0 = unifrnd(V0_range(1), V0_range(2));
    s0 = unifrnd(S0_range(1), S0_range(2));
    lambda_v = unifrnd(lambda_range(1), lambda_range(2)); % Assuming lambda_v and lambda_s share the same range
    lambda_s = unifrnd(lambda_range(1), lambda_range(2));
    omega = unifrnd(omega_range(1), omega_range(2));
    error1 = unifrnd(error_range(1), error_range(2));
    error2 = unifrnd(error_range(1), error_range(2));
    error3 = unifrnd(error_range(1), error_range(2));
    
    % File name for saving
    fname = fullfile(sim_folder, sprintf('sim_%d.mat', sim_count));

    % Store changing parameters
    params = struct('v0', v0, 's0', s0, ...
                    'lambda_v', lambda_v, 'lambda_s', lambda_s, ...
                    'omega', omega, 'error1', error1, 'error2', error2, 'error3', error3);

    % Initialize parameters
    parameters = struct('nparticles', 100, 'x0_unc', 1, ...
                        'lambda_v', lambda_v, 'lambda_s', lambda_s, ...
                        'v0', v0, 's0', s0, 'omega', omega);

    config = struct('rng_id', sim_count, 'nsim', 250, 'model_parameters', parameters);
    rng(config.rng_id);

    % Allocate memory for outputs
    nsim = config.nsim;
    outcome = nan(N, nsim);
    vol = nan(N, nsim);
    stc = nan(N, nsim);
    lr = nan(N, nsim);
    val = nan(N, nsim);
    biased_state = nan(N, nsim); % Initialize biased_state

    % Run trials
    for i = 1:nsim
        [observed_trial, biased_trial] = timeseries(base_x, omega, error1, error2, error3);
        outcome(:, i) = observed_trial; %Noise and Biaspresent
        biased_state(:, i) = biased_trial; %Only Bias present
        [vol(:, i), stc(:, i), lr(:, i), val(:, i)] = model_pf(outcome(:, i), config.model_parameters);
    end

    % Store results
    sim = struct('config', config, 'val', val, 'vol', vol, 'stc', stc, ...
                 'lr', lr, 'outcome', outcome,  'params', params, 'observed', outcome, 'biased_state', biased_state);
    save(fname, 'sim');

    % Plot and Save the Simulation
    % plot_and_save(sim, sim_count, plot_folder, base_x); % Uncomment if you want to generate plots

    % Increment count
    sim_count = sim_count + 1;
end
end

%% Function to Load Data, Plot, and Save Figure
function plot_and_save(sim, sim_idx, plot_folder, base_x)
% Extract simulation data
vol = mean(sim.vol, 2);
stc = mean(sim.stc, 2);
val = mean(sim.val, 2);
e_vol = serr(sim.vol, 2);
e_stc = serr(sim.stc, 2);
e_val = serr(sim.val, 2);
params = sim.params;
observed = mean(sim.observed, 2); % Observed data (o)
% Labels for subplots
xstr = {'Observed Data', 'Volatility', 'Stochasticity'};
% Create new figure
figure('Visible', 'off'); 
set(gcf, 'units', 'normalized', 'position', [0 0 .9 1]);
    
% Plot Estimated Reward
subplot(3, 2, 1);
plot_signal(3, 2, 1, {val}, {e_val}, {'Estimated reward'}, [], [], [], [], [0 0 0]);
hold on;
plot(val, '--k', 'linewidth', 1);
%ylim([-20 20]);
% Plot Observed Data Instead of Learning Rate Bar Chart
subplot(3, 2, 2);
plot(1:length(observed), observed, 'b-', 'LineWidth', 1.5);
title('Observed Original Data');
xlabel('Time Steps');
ylabel('Value');
lr = mean(sim.lr, 2);       % Average learning rate across simulations
e_lr = serr(sim.lr, 2);     % Standard error
% For example, insert this before saving:
subplot(3, 2, 5);  % Additional subplot position
plot_signal(3, 2, 5, {lr}, {e_lr}, {'Learning Rate'}, [], [], [], [], [0 0 0]);
% Plot Volatility
subplot(3, 2, 3);
plot_signal(3, 2, 3, {vol}, {e_vol}, xstr(2), [], [], [], [], [0 0 0]);
% Plot Stochasticity
subplot(3, 2, 4);
plot_signal(3, 2, 4, {stc}, {e_stc}, xstr(3), [], [], [], [], [0 0 0]);
% Title with Parameter Values
sgtitle(sprintf('Sim %d: v0=%.2f, s0=%.4f, λ_v=%.2f, λ_s=%.2f, ω=%.3f, error1=%d, error2=%d, error3=%d', ...
                sim_idx, params.v0, params.s0, params.lambda_v, params.lambda_s, params.omega, params.error1, params.error2, params.error3));
% Save figure
plot_filename = fullfile(plot_folder, sprintf('simulation_%d.png', sim_idx));
saveas(gcf, plot_filename); % Save as PNG
% Close to free memory
close(gcf);
end
%% Function to Read Observed Data
function [y,x]=timeseries(base_x, omega, error1, error2, error3)
x = base_x;
N = size(x, 1);
error1_ranges = [0, 40; 248, 285; 172, 194; 353, 377]; % 3.6 - - + +
error2_ranges = [134, 172; 222, 248; 40, 61; 314, 353]; % 7.2 - - + + 
error3_ranges = [97, 134; 285, 314; 61, 97; 194, 222]; % 14.4 - - + +
for i = 1:size(error1_ranges, 1)
    idx = (error1_ranges(i,1)+1):(error1_ranges(i,2));
    if i <= 2
        x(idx) = x(idx) + error1;  % first two ranges → add
    else
        x(idx) = x(idx) - error1;  % last two ranges → subtract
    end
end
for i = 1:size(error2_ranges, 1)
    idx = (error2_ranges(i,1)+1):(error2_ranges(i,2));
    if i <= 2
        x(idx) = x(idx) + error2;  % first two ranges → add
    else
        x(idx) = x(idx) - error2;  % last two ranges → subtract
    end
end
for i = 1:size(error3_ranges, 1)
    idx = (error3_ranges(i,1)+1):(error3_ranges(i,2));
    if i <= 2
        x(idx) = x(idx) + error3;  % first two ranges → add
    else
        x(idx) = x(idx) - error3;  % last two ranges → subtract
    end
end
y = x + sqrt(omega)*randn(N,1);
end