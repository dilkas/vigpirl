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
gp.rbf_var = random('Chisquare', algorithm_params.rbf_init);
%gp.rbf_var = 1; % TEMP
gp.inv_widths = random('Chisquare', algorithm_params.ard_init,...
    [1, size(feature_data.splittable, 2)]);
%gp.inv_widths = ones(1, size(feature_data.splittable, 2));