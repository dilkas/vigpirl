function answer = estimate_derivative(u, matrices, r_covariance_matrix,...
  mdp_data, Sigma, mu, mdp_model, example_samples, counts, T, B)
             % Uses a point-sample of u to estimate the part of dL/dnu under E[]

  function answer = lambda_derivative(i)
    % Returns the derivative of the ELBO w.r.t. lambda_i.
    % (Actually, the part of the derivative inside E[] without v)
    R = S * matrices.Kru_grad(:, :, i) - matrices.Krr_grad(:, :, i) +...
      (matrices.Kru_grad(:, :, i)' - S * matrices.Kuu_grad(:, :, i)) * Kuu_inv * matrices.Kru;
    t = r' - S * u';
    answer = trace(R * adjoint(Gamma)) / det(Gamma) - t' * Gamma_inv * R * Gamma_inv * t;
  end

  % The part of v specific to a single state-action pair
  function elbo_part = expected_derivative_for(state_action)
    state = state_action(1);
    action = state_action(2);
    next_states = extract(mdp_data.sa_s, state, action);
    next_probabilities = extract(mdp_data.sa_p, state, action);
    s = next_probabilities' * solution.v(next_states, 1);
    elbo_part = solution.v(state, 1) - mdp_data.discount * s;
  end

  Kuu_inv = inv(matrices.Kuu);
  Sigma_inv = inv(Sigma);
  U = (u - mu) * (u - mu)';
  S = matrices.Kru' * Kuu_inv;
  Gamma = matrices.Krr - S * matrices.Kru;
  Gamma_inv = inv(Gamma);
  %r = mvnrnd(S * u', r_covariance_matrix);
  r = (S * u')';
  solution = feval([mdp_model 'solve'], mdp_data, r');
  v_for_each_state_action = cellfun(@expected_derivative_for,...
    example_samples, 'Uniform', 0);
  v = sum(cat(3, v_for_each_state_action{:}), 3);

  unique_lambda_part = arrayfun(@lambda_derivative, 1:size(matrices.Kuu_grad, 3));
  unique_mu_part = (u' - mu)' * (Sigma_inv + Sigma_inv');
  unique_B_part = 2*(Sigma_inv*U*Sigma_inv - adjoint(Sigma)/det(Sigma)) * B;

  unique_part = vertcat(2, unique_lambda_part', diag(unique_B_part),...
    unique_mu_part', get_lower_triangle(unique_B_part));
  answer = unique_part * v;
end

function a = extract(mdp, state, action)
  temp = mdp(state, action, :);
  a = reshape(temp, size(mdp, 3), 1);
end
