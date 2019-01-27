function answer = estimate_derivative(u, Krr, Kru, Kuu, Kuu_inv, r_covariance_matrix,...
                                      Kuu_derivatives, Kru_derivatives, Krr_derivatives, mdp_data, Sigma, Sigma_inv,...
                                      mu, mdp_model, example_samples, counts, T, B)
             % Uses a point-sample of u to estimate the part of dL/dnu under E[]

  function answer = lambda_derivative(i)
    % Returns the derivative of the ELBO w.r.t. lambda_i.
    % (Actually, the part of the derivative inside E[] without v)

    Rr = Ss * Kru_derivatives(:, :, i) - Krr_derivatives(:, :, i) +...
      (Kru_derivatives(:, :, i)' - Ss * Kuu_derivatives(:, :, i)) * Kuu_inv * Kru;
    t = r' - Ss * u';
    answer = trace(Rr * adjoint(Gamma)) / det(Gamma) - t' * Gamma_inv * Rr * Gamma_inv * t;
  end

  function answer = expected_derivative_for(state_action)
    % TODO: check unique_B_part. Doesn't match the PDF. Update the PDF?
    state = state_action(1);
    action = state_action(2);
    next_states = extract(mdp_data.sa_s, state, action);
    next_probabilities = extract(mdp_data.sa_p, state, action);

    unique_lambda_part = arrayfun(@lambda_derivative, 1:size(Kuu_derivatives, 3));
    unique_mu_part = (u' - mu)' * (Sigma_inv + Sigma_inv');
    %unique_B_part = ((adjoint(T) + adjoint(Sigma)) / det(Sigma) + T*U*T +...
    %  Sigma_inv*U*Sigma_inv) * B;
    unique_B_part = Sigma_inv*U*Sigma_inv*B - adjoint(Sigma)*B/det(Sigma);

    unique_part = vertcat(2, unique_lambda_part', diag(unique_B_part),...
      unique_mu_part', get_lower_triangle(unique_B_part));
    s = next_probabilities' * solution.v(next_states, 1);
    elbo_part = solution.v(state, 1) - mdp_data.discount * s;
    answer = unique_part * elbo_part;
  end

  U = (u - mu) * (u - mu)';
  Ss = Kru' * Kuu_inv;
  Gamma = Krr - Ss * Kru;
  Gamma_inv = inv(Gamma);
  %r = mvnrnd(Ss * u', r_covariance_matrix);
  r = (Ss * u')';
  solution = feval([mdp_model 'solve'], mdp_data, r');
  derivatives_for_each_state_action = cellfun(@expected_derivative_for,...
    example_samples, 'Uniform', 0);
  answer = sum(cat(3, derivatives_for_each_state_action{:}), 3);
end

function a = extract(mdp, state, action)
  temp = mdp(state, action, :);
  a = reshape(temp, size(mdp, 3), 1);
end
