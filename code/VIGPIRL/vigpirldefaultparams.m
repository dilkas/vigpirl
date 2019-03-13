% Fill in default parameters for the GPIRL algorithm.
function algorithm_params = vigpirldefaultparams(algorithm_params)

% Create default parameters.
default_params = struct(...
    ...% These are initial values.
    'ard_init',1,...
    'noise_init',0.01,...
    'rbf_init',5,...
    ...% These parameters control how the inducing points are selected.
    'inducing_pts','examplesplus',...
    'inducing_pts_count',16,...
    'samples_count',100,...
    'learning_rate',0.1,...
    'B_learning_rate',0.001,...
    'num_iterations',20,...
    'required_precision', 0.001);

% Set parameters.
algorithm_params = filldefaultparams(algorithm_params,default_params);
