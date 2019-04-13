% Fill in default parameters for the VIGPIRL algorithm.
function algorithm_params = vigpirldefaultparams(algorithm_params)

% Create default parameters.
default_params = struct(...
    'ard_init', 1,... % Mean initial value for lambda1
    'noise_init', 0.01,... % 2 * sigma^2, the noise factor
    'rbf_init', 5,... % Mean initial value for lambda0
    'inducing_pts', 'examplesplus',...
    'inducing_pts_count', 16,... % Number of inducing points
    'samples_count', 100,... % S, the number of samples
    'learning_rate', 0.1,... % Step sizes
    'lambda1_learning_rate', 0.1,...
    'B_learning_rate', 0.1,...
    'num_iterations', 10,... % Max number of iterations for optimisation
    'required_precision', 0.01,... % Alternatively, tolerance for convergence
    'random_initial_B', false,...
    'disable_optimising_lambda0', false,... % Mostly for debugging
    'disable_optimising_lambda', false,...
    'disable_optimising_mu', false,...
    'disable_optimising_B_diagonal', false,...
    'disable_optimising_rest_of_B', false,...
    'step_size_algorithm', 'constant'... % Other values are 'AdaGrad' and 'AdaDelta'
);

% Set parameters.
algorithm_params = filldefaultparams(algorithm_params, default_params);
