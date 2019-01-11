                                % GP-based non-linear IRL algorithm.
function irl_result = vigpirlrun(algorithm_params,mdp_data,mdp_model,...
                                 feature_data,example_samples,~,~)

                    % algorithm_params - parameters of the GP IRL algorithm.
                    % mdp_data - definition of the MDP to be solved.
                    % example_samples - cell array containing examples.
                    % irl_result - result of IRL algorithm (see bottom of file).

                                % Fill in default parameters.
  algorithm_params = vigpirldefaultparams(algorithm_params);

                      % Get state-action counts and initial state distributions.
  [mu_sa, counts] = vigpirlgetstatistics(example_samples, mdp_data);

                                % Create initial GP.
  gp = vigpirlinit(algorithm_params, feature_data);
                                % Choose inducing points.
  %gp = vigpirlgetinducingpoints(gp,mu_sa,algorithm_params);
  gp.X_u = gp.X; % TEMP
  m = size(gp.X_u, 1);
  gp.mu = rand(1, m)';

                 % B is a lower triangular matrix with positive diagonal entries
  gp.B = normrnd(0, 1, [m, m]);
  gp.B(1:m+1:end) = random('Chisquare', 4, [m, 1]);
  gp.B = tril(gp.B);
  %gp.B = eye(m); % TEMP
  %gp.D = diag(normrnd(0, 1));
  gp.D = zeros(m, m); % TEMP
  % TODO: B could be m x p for p << m
  % diag() might not work in that case, and need to update the initialisation

  d = size(feature_data.splittable, 2);
  elbo_list = [];

  % for AdaGrad
  G = zeros(m + d + 1 + m*(m+1)/2, 1);
  eta = 0.1;

  % for AdaDelta
  %num_hyperparameters = m + d + 1 + m*(m+1)/2;
  %E_g = zeros(num_hyperparameters, 1);
  %E_x = zeros(num_hyperparameters, 1);
  %epsilon = 1e-6;
  %rho = 0.1;

  tic;
  i = 0;
  while true
    % Compute the gradient
    [Kru, Kuu, Kuu_inv, KruKuu, Krr, Kuu_grad, Kru_grad] = vigpirlkernel(gp);
    Sigma = gp.B * gp.B' + gp.D * gp.D;
    Sigma_inv = inv(Sigma);
    z = mvnrnd(gp.mu', Sigma, algorithm_params.samples_count);
    T = inv(gp.B * gp.B' + (gp.D * gp.D)');
    full_grad = full_gradient(Sigma, Sigma_inv, mdp_model, mdp_data,...
      example_samples, counts, gp.mu, Kru, Kuu, Kuu_inv, KruKuu, Krr, Kuu_grad,...
      Kru_grad, z, T, gp.B);

    old_hyperparameters = vigpirlpackparam(gp);
    grad = transform_gradient(full_grad(2:end), old_hyperparameters, d, m);

    %grad(1:d+1) = 0; % lambda
    %grad(d+m+2:d+2*m+1) = 0; % mu
    %grad(d+2:d+m+1) = 0; % B diagonal
    %grad(d+2*m+2:end) = 0; % rest of B

    fprintf('Hyperparameters:\n');
    %disp(old_hyperparameters(d+m+2:d+2*m+1));
    disp(old_hyperparameters);
    fprintf('Gradient:\n');
    %disp(grad(d+m+2:d+2*m+1));
    disp(grad);

    % for AdaGrad
    G = G + grad .^ 2;
    rho = (eta / sqrt(G))';

    % for AdaDelta
    %E_g = rho * E_g + (1 - rho) * grad .^ 2;
    %delta = sqrt(E_x + epsilon) ./ sqrt(E_g + epsilon) .* grad;
    %E_x = rho * E_x + (1 - rho) * delta .^ 2;

    %hyperparameters = old_hyperparameters + 10 * delta;
    %hyperparameters = old_hyperparameters + eta * grad;
    hyperparameters = old_hyperparameters + rho .* grad;
    gp = vigpirlunpackparam(gp, hyperparameters);

    %fprintf('New hyperparameters:\n');
    %disp(hyperparameters(d+m+2:d+2*m+1));
    disp(full_grad(1));
    fprintf('----------\n');
    elbo_list = horzcat(elbo_list, full_grad(1));

    %if norm(hyperparameters - old_hyperparameters, 1) < 0.001
    %  break;
    %end

    i = i + 1;
    if (i > 100)
      break;
    end
  end
  time = toc;
  stem(elbo_list);

                                % Return corresponding reward function.
  r = KruKuu * gp.mu;
  solution = feval([mdp_model 'solve'], mdp_data, r);
  v = solution.v;
  q = solution.q;
  p = solution.p;

                                % Construct returned structure.
  irl_result = struct('r', r, 'v', v, 'p', p, 'q', q, 'model_itr', {{gp}},...
                      'r_itr', {{r}}, 'model_r_itr', {{r}}, 'p_itr', {{p}},...
                      'model_p_itr', {{p}}, 'time', time, 'score', 0);
end

function new_grad = transform_gradient(grad, hyperparameters, d, m)
  num_transformed = 1 + d + m;
  new_grad = vertcat(vigpirlhpxform(hyperparameters(1:num_transformed),...
    grad(1:num_transformed), 'exp', 2), grad(num_transformed+1:end));
end