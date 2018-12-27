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
  gp = gpirlinit(algorithm_params,feature_data);
                              % Choose inducing points.
  gp = vigpirlgetinducingpoints(gp,mu_sa,algorithm_params);

  m = size(gp.X_u, 1);

                                % Randomly initialize variational parameters
                                % TODO: do the optimal thing instead
  mu = rand(m, 1);

                 % L is a lower triangular matrix with positive diagonal entries
  L = normrnd(0, 1, [m, m]);
  L(1:m+1:end) = random('Chisquare', 1, [m, 1]);
  L = tril(L);

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
      trace_part = arrayfun(@(i) trace((Kuu_inv * (u' * u) * Kuu_inv' -...
        Kuu_inv) * Kuu_derivatives(:, :, i)), 1:size(Kuu_derivatives, 3));
      s = next_probabilities' * solution.v(next_states, 1);
      answer = trace_part .* (solution.v(state, 1) - mdp_data.discount * s);
    end

    r = mvnrnd(KruKuu * u', Krr - KruKuu * Kru);
    solution = feval([mdp_model 'solve'], mdp_data, r');
    derivatives_for_each_state_action = cellfun(@expected_derivative_for,...
      example_samples, 'Uniform', 0);
    answer = sum(cat(3, derivatives_for_each_state_action{:}), 3);
  end

  % FIXME: Kru_derivative_lambda is not used. Replace Kuu_derivate_lambda with pairs of derivatives
  tic;
  log_rbf_var = log(gp.rbf_var);
  log_lambda = log(gp.inv_widths);
  for i = 1:10
                 % Draw samples_count samples from the variational approximation
    disp(log_rbf_var);
    rho = 0.001; % TODO: implement AdaGrad from BBVI
    [Kru, Kuu_inv, KruKuu, Krr, Kuu_derivative, Kuu_derivative_lambda,...
      Kru_derivative, Kru_derivative_lambda] = vigpirlkernel(gp);
    z = mvnrnd(mu, L * L', algorithm_params.samples_count);

    %estimate = mean(arrayfun(@(row_id)...
    %  estimate_derivative(z(row_id, :)', KruKuu, Krr, Kru, Kuu_inv,...
    %  Kuu_derivative), (1:size(z, 1)).'));
    derivatives_for_each_u = arrayfun(@(i) estimate_derivative(z(i, :),...
      KruKuu, Krr, Kru, Kuu_inv, Kuu_derivative_lambda), 1:size(z, 1), 'Uniform', 0);
    estimates = mean(cat(3, derivatives_for_each_u{:}), 3);

    %log_rbf_var = log_rbf_var + rho * gp.rbf_var * (-0.5 * trace(Kuu_inv * Kuu_derivative)...
    %                           + init_s' * (Kru_derivative' - KruKuu * Kuu_derivative)...
    %                             * Kuu_inv * mu - 0.5 * estimate);
    log_lambda = log_lambda - 0.5 * rho * gp.inv_widths .*...
      arrayfun(@(i) trace(Kuu_inv * Kuu_derivative_lambda(:, :, i)) + init_s' *...
      (Kru_derivative' - KruKuu * Kuu_derivative_lambda(:, :, i)) * Kuu_inv *...
      mu, 1:size(Kuu_derivative_lambda, 3)) - 0.5 * rho * gp.inv_widths .* estimates;
    gp.rbf_var = exp(log_rbf_var);
    gp.inv_widths = exp(log_lambda);
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
