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

                 % L is a lower triangular matrix with positive diagonal entries
  L = normrnd(0, 1, [m, m]);
  L(1:m+1:end) = random('Chisquare', 4, [m, 1]);
  L = tril(L);
  Sigma = L * L';

  tic;
  for n = 1:10
                 % Draw samples_count samples from the variational approximation
    rho = 1/n; % TODO: implement AdaGrad from BBVI
    [Kru, Kuu_inv, KruKuu, Krr, Kuu_grad, Kru_grad] = vigpirlkernel(gp);
    z = mvnrnd(mu, Sigma, algorithm_params.samples_count);
    changes = full_gradient(Sigma, mdp_model, mdp_data, example_samples,...
      counts, mu, Kru, Kuu_inv, KruKuu, Krr, Kuu_grad, Kru_grad, z);
    hyperparameters = vigpirlpackparam(gp);
    hyperparameters = hyperparameters + rho *...
      gpirlhpxform(hyperparameters, changes, 'exp', 2);
    gp = vigpirlunpackparam(gp, hyperparameters);
  end
  time = toc;
                                % TODO: stopping condition

                                % Return corresponding reward function.
  r = KruKuu * gp.mu;
  solution = feval([mdp_model 'solve'], mdp_data, r);
  v = solution.v;
  q = solution.q;
  p = solution.p;

                                % Construct returned structure.
                                % TODO: score = best elbo
  irl_result = struct('r', r, 'v', v, 'p', p, 'q', q, 'model_itr', {{gp}},...
                      'r_itr', {{r}}, 'model_r_itr', {{r}}, 'p_itr', {{p}},...
                      'model_p_itr', {{p}}, 'time', time, 'score', 0);
end
