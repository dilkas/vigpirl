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
mdp_data = struct('discount', 0.9, 'states', 3, 'actions', 2);
mdp_data.sa_s(:, :, 1) = [2, 3; 1, 3; 1, 2];
mdp_data.sa_p(1:3, 1:2, 1) = 1;
feature_data = struct('splittable', [1; 2; 3]);
example_samples = {[1, 1], [3, 2]};
wrapper(mdp_data, feature_data, example_samples);
%vigpirlrun(struct(), mdp_data, 'linearmdp', feature_data, example_samples);
return;

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
max_demonstrations_count = 3;
num_repeats = 10;
data = [];
data2 = [];
data3 = [];
data4 = [];
for x = 1:max_demonstrations_count
    for y = 1:max_demonstrations_count
        fprintf('x=%d, y=%d\n', x, y);
        demonstrations = {};
        [demonstrations{1:x}] = deal([1, 1]);
        [demonstrations{x+1:x+y}] = deal([3, 2]);
        temp_data1 = [];
        temp_data2 = [];
        for i = 1:num_repeats
            result = wrapper(mdp_data, feature_data, demonstrations);
            S = result.matrices.Kru' * inv(result.matrices.Kuu);
            Gamma = result.matrices.Krr - S * result.matrices.Kru;
            covariance = Gamma + S * result.gp.B * result.gp.B' * S';
            % The format is: first column top to bottom, then second, etc.
            temp_data1 = [temp_data1; reshape(covariance, [1, 9])];
            temp_data2 = [temp_data2; reshape(Gamma, [1, 9])];
        end
        data = [data; [x, y, mean(temp_data1, 1)]];
        data2 = [data2; [x, y, median(temp_data1, 1)]];
        data3 = [data3; [x, y, mean(temp_data2, 1)]];
        data4 = [data4; [x, y, median(temp_data2, 1)]];
    end
end
writematrix(data, '../mpaper/mean_covariance.csv');
writematrix(data2, '../mpaper/median_covariance.csv');
writematrix(data3, '../mpaper/mean_Gamma.csv');
writematrix(data4, '../mpaper/median_Gamma.csv');

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