% Initialize an empty table
final_table = [];

% Load raw wind condition data
raw_data = readmatrix('Wind_Conditions.xlsx'); 
raw_data = raw_data(:,1) 
raw_data = raw_data(81:end);
scaled_data = raw_data * 10000; % Scale the data

% Loop through sim_1.mat to sim_400.mat
for agent_id = 0:4999
    % Load the simulation data
    filename = sprintf('sim_%d.mat', agent_id + 1); % Files are named sim_1, sim_2, ..., sim_25
    sim_data = load(filename); 
    sim = sim_data.sim;

    % Compute averages
    avg_val = mean(sim.val, 2);
    avg_volatility = mean(sim.vol, 2);
    avg_stochasity = mean(sim.stc, 2);
    trials = (1:length(avg_val))'; % Assuming trials are sequential (1 to N)
    avg_lr = mean(sim.lr, 2);

    % Create empty values for missing columns
    empty_col = repmat({''}, size(avg_val));

    % Create a table for this simulation
    T = table(repmat(agent_id, size(avg_val)), avg_val, empty_col, ...
        scaled_data, empty_col, empty_col, ...
        trials, avg_volatility, avg_stochasity, empty_col, ...
        empty_col, empty_col, empty_col, ...
        'VariableNames', {'agentid', 'actions', 'correct_actions', 'rewards', 'isswitch', ...
        'iscorrectaction', 'trials', 'rpe_history', 'unchosen_rpe_history', 'alpha', 'beta', ...
        'neg_alpha', 'stickiness'});

    % Append to the final table
    final_table = [final_table; T];
end

% Save the final combined table to CSV
writetable(final_table, 'Combined_Simulations_More_Lambda.csv');

disp('CSV file generated: combined_simulations.csv');
