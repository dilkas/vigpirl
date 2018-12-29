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
  [mu_sa,init_s] = vigpirlgetstatistics(example_samples,mdp_data);

                              % Create initial GP.
  gp = vigpirlinit(algorithm_params,feature_data);
                              % Choose inducing points.
  gp = vigpirlgetinducingpoints(gp,mu_sa,algorithm_params);

  m = size(gp.X_u, 1);

                 % L is a lower triangular matrix with positive diagonal entries
  L = normrnd(0, 1, [m, m]);
  L(1:m+1:end) = random('Chisquare', 1, [m, 1]);
  L = tril(L);
  Sigma = L * L';
  Sigma_inv = inv(Sigma);

  function a = extract(mdp, state, action)
    temp = mdp(state, action, :);
    a = reshape(temp, size(mdp, 3), 1);
  end

  function answer = estimate_derivative(u, KruKuu, Krr, Kru, Kuu_inv, Kuu_derivatives)
    function answer = expected_derivative_for(state_action)
      state = state_action(1);
      action = state_action(2);
      next_states = extract(mdp_data.sa_s, state, action);
      next_probabilities = extract(mdp_data.sa_p, state, action);
      unique_lambda_part = arrayfun(@(i) trace((Kuu_inv * (u' * u) * Kuu_inv' -...
        Kuu_inv) * Kuu_derivatives(:, :, i)), 1:size(Kuu_derivatives, 3));
      unique_mu_part = (Sigma_inv + Sigma_inv') * (u - gp.mu)';
      unique_part = vertcat(unique_lambda_part', unique_mu_part);
      s = next_probabilities' * solution.v(next_states, 1);
      answer = unique_part * (solution.v(state, 1) - mdp_data.discount * s);
    end

    r = mvnrnd(KruKuu * u', Krr - KruKuu * Kru);
    solution = feval([mdp_model 'solve'], mdp_data, r');
    derivatives_for_each_state_action = cellfun(@expected_derivative_for,...
      example_samples, 'Uniform', 0);
    answer = sum(cat(3, derivatives_for_each_state_action{:}), 3);
  end

  tic;
  for i = 1:10
                 % Draw samples_count samples from the variational approximation
    rho = 0.001; % TODO: implement AdaGrad from BBVI
    [Kru, Kuu_inv, KruKuu, Krr, Kuu_grad, Kru_grad] = vigpirlkernel(gp);
    z = mvnrnd(gp.mu, Sigma, algorithm_params.samples_count);

    grad_prediction_for_each_u = arrayfun(@(i) estimate_derivative(z(i, :),...
      KruKuu, Krr, Kru, Kuu_inv, Kuu_grad), 1:size(z, 1), 'Uniform', 0);
    estimated_grad = mean(cat(3, grad_prediction_for_each_u{:}), 3);

    not_estimated_lambda = arrayfun(@(i) -0.5 *...
      trace(Kuu_inv * Kuu_grad(:, :, i)) + init_s' * (Kru_grad(:, :, i)' -...
      KruKuu * Kuu_grad(:, :, i)) * Kuu_inv * gp.mu', 1:size(Kuu_grad, 3));
    not_estimated_mu = init_s' * Kru' * Kuu_inv;
    not_estimated = vertcat(not_estimated_lambda', not_estimated_mu');
    changes = not_estimated - 0.5 * estimated_grad;

    hyperparameters = vigpirlpackparam(gp);
    disp(hyperparameters);
    fprintf('-----');
    hyperparameters = hyperparameters + rho *...
      gpirlhpxform(hyperparameters, changes, 'exp', 2);
    gp = vigpirlunpackparam(gp, hyperparameters);
  end

  time = toc;
                                % TODO: stopping condition

                                % Return corresponding reward function.
  r = KruKuu * mu;
  solution = feval([mdp_model 'solve'], mdp_data, r);
  v = solution.v;
  q = solution.q;
  p = solution.p;

                                % Construct returned structure.
                                % TODO: score = best elbo
  irl_result = struct('r',r,'v',v,'p',p,'q',q,'model_itr',{{gp}},...
                      'r_itr',{{r}},'model_r_itr',{{r}},'p_itr',{{p}},'model_p_itr',{{p}},...
                      'time',time,'score', 0);
end
