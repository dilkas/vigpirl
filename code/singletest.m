% Convenience script for running a single test.
addpaths;

% GPIRL
%test_result = runtest('gpirl',struct(),'linearmdp',...
%    'objectworld',struct('n',32,'determinism',0.7,'seed',1,'continuous',0),...
%    struct('training_sample_lengths',8,'training_samples',16,'verbosity',2));

% DGP-IRL original
%test_result = runtest('deepgpirl',struct(),'linearmdp','binaryworld',...
%   struct('n',12),struct('training_sample_lengths',12^2,...
%  'training_samples',8,'verbosity',1));

% GPIRL's test using DGP-IRL
%test_result = runtest('deepgpirl',struct('R', 4),'linearmdp','objectworld',...
%   struct('n',12, 'continuous', 1),...
%   struct('training_sample_lengths',12^2,...
%   'training_samples',64,'verbosity',1));

% VIGPIRL
%test_result = runtest('vigpirl',struct(),'linearmdp',...
%    'gridworld',struct('n',8,'determinism',1,'seed',1,'continuous',0),...
%    struct('training_sample_lengths',32,'training_samples',16,'verbosity',2));

% Visualize solution.
%printresult(test_result);
%visualize(test_result);

% Two states
%mdp_data = struct('sa_s', [2; 1], 'sa_p', [1; 1], 'discount', 0, 'states', 2, 'actions', 1);
%feature_data = struct('splittable', [1; 2]);
%example_samples = {[1, 1]};

% Three states
%mdp_data = struct('discount', 0.9, 'states', 3, 'actions', 2);
%mdp_data.sa_s(:, :, 1) = [2, 3; 1, 3; 1, 2];
%mdp_data.sa_p(1:3, 1:2, 1) = 1;
%feature_data = struct('splittable', [1; 2; 3]);
%example_samples = {[1, 1], [3, 2]};
%wrapper(mdp_data, feature_data, example_samples);
%vigpirlrun(struct(), mdp_data, 'linearmdp', feature_data, example_samples);
%return;

% Fancy stuff 1
%max_demonstrations_count = 3;
%num_repeats = 10;
%data = [];
%for x = 1:max_demonstrations_count
%    for y = 1:max_demonstrations_count
%        fprintf('x=%d, y=%d\n', x, y);
%        demonstrations = {};
%        [demonstrations{1:x}] = deal([1, 1]);
%        [demonstrations{x+1:x+y}] = deal([3, 2]);
%        temp_data1 = [];
%        for i = 1:num_repeats
%            result = wrapper(mdp_data, feature_data, demonstrations);
            % The format is: first column top to bottom, then second, etc.
%            temp_data1 = [temp_data1; reshape(result.gp.B * result.gp.B', [1, 9]), reshape(result.p, [1, 6])];
%       end
%       data = [data; [x, y, mean(temp_data1, 1)]];
%    end
%end
%writematrix(data, '../mpaper/covariance_and_policy.csv');

% Fancy stuff 2
%max_demonstrations_count = 3;
%num_repeats = 10;
%data = [];
%data2 = [];
%data3 = [];
%data4 = [];
%for x = 1:max_demonstrations_count
%    for y = 1:max_demonstrations_count
%        fprintf('x=%d, y=%d\n', x, y);
%        demonstrations = {};
%        [demonstrations{1:x}] = deal([1, 1]);
%        [demonstrations{x+1:x+y}] = deal([3, 2]);
%        temp_data1 = [];
%        temp_data2 = [];
%        for i = 1:num_repeats
%            result = wrapper(mdp_data, feature_data, demonstrations);
            % The format is: first column top to bottom, then second, etc.
%            temp_data1 = [temp_data1; reshape(covariance_matrix(result), [1, 9])];
%            temp_data2 = [temp_data2; reshape(Gamma, [1, 9])];
%        end
%        data = [data; [x, y, mean(temp_data1, 1)]];
%        data2 = [data2; [x, y, median(temp_data1, 1)]];
%        data3 = [data3; [x, y, mean(temp_data2, 1)]];
%        data4 = [data4; [x, y, median(temp_data2, 1)]];
%    end
%end
%writematrix(data, '../mpaper/mean_covariance.csv');
%writematrix(data2, '../mpaper/median_covariance.csv');
%writematrix(data3, '../mpaper/mean_Gamma.csv');
%writematrix(data4, '../mpaper/median_Gamma.csv');

% Fancy stuff 3: the clique edition
n = 10;
num_demonstrations = 100;
num_repeats = 100;

mdp_data = struct('discount', 0.9, 'states', n, 'actions', n - 1);
for i = 1:n
    mdp_data.sa_s(i, :, 1) = horzcat(1:(i - 1), (i + 1):n);
end
mdp_data.sa_p(1:n, 1:(n - 1), 1) = 1;
feature_data = struct('splittable', (1:n)'); % could do a one-hot encoding instead

total_values = [];
total_groups = [];
for i = 1:num_repeats
    [random_demonstrations, targeted_demonstrations] = generate_demonstrations(num_demonstrations, n)
    random_result = wrapper(mdp_data, feature_data, random_demonstrations);
    targeted_result = wrapper(mdp_data, feature_data, targeted_demonstrations);
    random_covariance = covariance_matrix(random_result);
    targeted_covariance = covariance_matrix(targeted_result);
    [values, groups] = meaningful_classes(abs(random_covariance) - abs(targeted_covariance));
    total_values = [total_values, values];
    total_groups = [total_groups, groups];
end

figure('Units', 'centimeters', 'Position', [0 0 12 8], 'PaperPositionMode', 'auto', 'Resize', 'off');
boxplot(total_values, total_groups);
ylabel('difference in covariance');
hold on;
refline([0 0]);
print('../mpaper/figures/boxplots', '-depsc2');

function [random_demonstrations, targeted_demonstrations] = generate_demonstrations(num_demonstrations, n)
    random_demonstrations = {};
    targeted_demonstrations = {};
    for i = 1:num_demonstrations
        random_demonstrations{i} = [randi(n), randi(n - 1)];
        targeted_demonstrations{i} = [1 + randi(n - 1), 1];
    end
end

function [values, groups] = meaningful_classes(matrix_of_differences)
    diagonal = diag(matrix_of_differences);
    the_rest = get_lower_triangle(matrix_of_differences);
    state = diagonal(1);
    other_diag = diagonal(2:end);
    last_neighbour_index = size(matrix_of_differences, 1) - 1;
    neighbours = the_rest(1:last_neighbour_index);
    non_neighbours = the_rest((last_neighbour_index + 1):end);

    values = horzcat(state, other_diag', neighbours', non_neighbours');
    groups = {};
    a = 1;
    b = length(state);
    [groups{a:b}] = deal('first state');
    a = b + 1;
    b = b + length(other_diag);
    [groups{a:b}] = deal('other states');
    a = b + 1;
    b = b + length(neighbours);
    [groups{a:b}] = deal('incident edges');
    a = b + 1;
    b = b + length(non_neighbours);
    [groups{a:b}] = deal('other edges');
    %groups = horzcat(zeros(1, length(state)), ones(1, length(other_diag)), 2 * ones(1, length(neighbours)), 3 * ones(1, length(non_neighbours)));
end

function covariance = covariance_matrix(result)
    S = result.matrices.Kru' * inv(result.matrices.Kuu);
    Gamma = result.matrices.Krr - S * result.matrices.Kru;
    covariance = Gamma + S * result.gp.B * result.gp.B' * S';
end

function result = wrapper(mdp_data, feature_data, demonstrations)
    warning('');
    try
        result = vigpirlrun(struct(), mdp_data, 'linearmdp', feature_data, demonstrations);
    catch
        fprintf('Failed!\n');
        result = wrapper(mdp_data, feature_data, demonstrations);
    end
    [warning_message, ~] = lastwarn;
    if ~isempty(warning_message)
        fprintf('Failed because of a warning!\n');
        result = wrapper(mdp_data, feature_data, demonstrations);
    end
end