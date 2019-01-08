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
  [mu_sa, counts] = vigpirlgetstatistics(example_samples,mdp_data);

                                % Create initial GP.
  gp = vigpirlinit(algorithm_params,feature_data);
                                % Choose inducing points.
  gp = vigpirlgetinducingpoints(gp,mu_sa,algorithm_params);
  m = size(gp.X_u, 1);
  gp.mu = rand(1, m)';

                 % B is a lower triangular matrix with positive diagonal entries
  gp.B = normrnd(0, 1, [m, m]);
  gp.B(1:m+1:end) = random('Chisquare', 4, [m, 1]);
  gp.B = tril(gp.B);
%  gp.D = diag(normrnd(0, 1));
  gp.D = zeros(m, m);
  % TODO: B could be m x p for p << m
  % TODO: diag() might not work in that case, and need to update the initialisation

  d = size(feature_data.splittable, 2);
  G = zeros(m + d + 1 + m*(m+1)/2, 1);
  eta = 1e2;

  tic;
  while true
                 % Draw samples_count samples from the variational approximation
    [Kru, Kuu, Kuu_inv, KruKuu, Krr, Kuu_grad, Kru_grad] = vigpirlkernel(gp);
    Sigma = gp.B * gp.B' + gp.D * gp.D;
    Sigma_inv = inv(Sigma);
    z = mvnrnd(gp.mu', Sigma, algorithm_params.samples_count);
    T = inv(gp.B * gp.B' + (gp.D * gp.D)');
    grad = full_gradient(Sigma, Sigma_inv, mdp_model, mdp_data,...
      example_samples, counts, gp.mu, Kru, Kuu, Kuu_inv, KruKuu, Krr, Kuu_grad,...
      Kru_grad, z, T, gp.B);
    changes = grad(2:end);
    G = G + changes .^ 2;
    rho = eta / sqrt(G);
    %disp(changes);
    fprintf('----------\n');
    old_hyperparameters = vigpirlpackparam(gp);
    hyperparameters = old_hyperparameters + rho' .*...
      vigpirlhpxform(old_hyperparameters, changes, 'exp', 2);
    gp = vigpirlunpackparam(gp, hyperparameters);

    if norm(hyperparameters - old_hyperparameters, 1) < 0.01
      break;
    end
  end
  time = toc;

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
