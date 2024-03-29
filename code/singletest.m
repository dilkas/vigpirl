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
%return;

mdp = ThreeStateMdp();

% One run, many plots
%model = Vigpirl(mdp, struct());
%model = model.run();
%model.convergence_plots();
%model.elbo_and_policy_convergence_plots();
%model.parameter_convergence_plot();

% Plots that run their own experiments
%model = Vigpirl(mdp, struct());
%model.elbo_contour_plot();
%model.policy_contour_plot();
%model.elbo_vs_gamma_plot();
%model.elbo_and_derivative_plot();

% Fancy experiments specific to the three-state MDP
%mdp.covariance_and_policy_with_more_data();
%mdp.covariances_with_more_data();

%clique_experiment();