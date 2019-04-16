% Initialize the GP structure for GPIRL based on feature data and algorithm
% parameters.
function gp = vigpirlinit(algorithm_params,feature_data)

% Create GP.
gp = struct();

% Initialize X.
gp.s_u = 1:size(feature_data.splittable,1);
gp.X = feature_data.splittable;
gp.X_u = feature_data.splittable;

% Initialize hyperparameters.
gp.noise_var = algorithm_params.noise_init;
gp.lambda0 = random('Chisquare', algorithm_params.lambda0_init);
%gp.lambda0 = 1;
gp.lambda = random('Chisquare', algorithm_params.lambda_init,...
    [1, size(feature_data.splittable, 2)]);
%gp.lambda = ones(1, size(feature_data.splittable, 2));