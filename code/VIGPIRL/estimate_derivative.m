function answer = estimate_derivative(u, KruKuu, Kuu, Kuu_inv, r_covariance_matrix,...
                                      Kuu_derivatives, mdp_data, Sigma, Sigma_inv,...
                                      mu, mdp_model, example_samples, counts, T, B)
             % Uses a point-sample of u to estimate the part of dL/dnu under E[]
  function answer = expected_derivative_for(state_action)
    state = state_action(1);
    action = state_action(2);
    next_states = extract(mdp_data.sa_s, state, action);
    next_probabilities = extract(mdp_data.sa_p, state, action);
    U = (u - mu) * (u - mu)';

    unique_lambda_part = arrayfun(@(i) trace((Kuu_inv * (u' * u) * Kuu_inv' -...
      Kuu_inv) * Kuu_derivatives(:, :, i)), 1:size(Kuu_derivatives, 3));
    unique_mu_part = (Sigma_inv + Sigma_inv') * (u' - mu);
    %unique_B_part = ((adjoint(T) + adjoint(Sigma)) / det(Sigma) + T*U*T +...
    %  Sigma_inv*U*Sigma_inv) * B;
    unique_B_part = Sigma_inv*U*Sigma_inv*B - adjoint(B*B')*B/det(Sigma);

    unique_part = vertcat(2, unique_lambda_part', unique_mu_part,...
      diag(unique_B_part), get_lower_triangle(unique_B_part));
    s = next_probabilities' * solution.v(next_states, 1);
    elbo_part = solution.v(state, 1) - mdp_data.discount * s;
    answer = unique_part * elbo_part;
  end

  %r = mvnrnd(KruKuu * u', r_covariance_matrix);
  r = (KruKuu * u')';
  solution = feval([mdp_model 'solve'], mdp_data, r');
  derivatives_for_each_state_action = cellfun(@expected_derivative_for,...
    example_samples, 'Uniform', 0);
  answer = sum(cat(3, derivatives_for_each_state_action{:}), 3);
end

function a = extract(mdp, state, action)
  temp = mdp(state, action, :);
  a = reshape(temp, size(mdp, 3), 1);
end
