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
test_result = runtest('vigpirl',struct(),'linearmdp',...
    'gridworld',struct('n',8,'determinism',0.7,'seed',1,'continuous',0),...
    struct('training_sample_lengths',8,'training_samples',16,'verbosity',2));

% Visualize solution.
%printresult(test_result);
visualize(test_result);

% Two states -- works well
%mdp_data = struct('sa_s', [2; 1], 'sa_p', [1; 1], 'discount', 0.9, 'states', 2, 'actions', 1);
%feature_data = struct('splittable', [1; 2]);
%example_samples = {[1, 1]};

% Three states -- doesn't work
%mdp_data = struct('discount', 0, 'states', 3, 'actions', 2);
%mdp_data.sa_s(:, :, 1) = [2, 3; 1, 3; 1, 2];
%mdp_data.sa_p(1:3, 1:2, 1) = 1;
%feature_data = struct('splittable', [1; 2; 3]);
%example_samples = {[1, 1], [3, 2]};
%vigpirlrun(struct(), mdp_data, 'linearmdp', feature_data, example_samples);
