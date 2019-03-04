function [elbo, grad] = full_gradient(mdp_data, demonstrations, counts, gp, z, matrices)
% TODO: will need to make this return optimal r as a function of mu so that
% vigpirlrun can construct an optimal solution (just like before)
% NOTE: returns the gradient for parameters in vector form

  function answer = estimate_derivative(u)
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
      elbo_part = solution.v(state, 1) + mdp_data.discount * s;
    end

    U = (u - gp.mu) * (u - gp.mu)';
    r = mvnrnd(S * u', Gamma);
    solution = linearmdpsolve(mdp_data, r');
    v_for_each_state_action = cellfun(@expected_derivative_for,...
      demonstrations, 'Uniform', 0);
    v = sum(cat(3, v_for_each_state_action{:}), 3);
    %fprintf('sample of v: %f\n', v);

    unique_lambda_part = arrayfun(@lambda_derivative, 1:size(matrices.Kuu_grad, 3));
    unique_mu_part = (u' - gp.mu)' * (Sigma_inv + Sigma_inv');
    unique_B_part = 2*(Sigma_inv*U*Sigma_inv - adjoint(Sigma)/det(Sigma)) * gp.B;

    unique_part = vertcat(2, unique_lambda_part', diag(unique_B_part),...
      unique_mu_part', get_lower_triangle(unique_B_part));
    answer = unique_part * v;
  end

  Sigma = gp.B * gp.B';
  Sigma_inv = inv(Sigma);
  Kuu_inv = inv(matrices.Kuu);
  S = matrices.Kru' * Kuu_inv;
  % we need to round Gamma because otherwise it's not symmetric due to numerical errors
  Gamma = round(matrices.Krr - S * matrices.Kru, 4);
  Gamma_inv = inv(Gamma);

  grad_prediction_for_each_u = arrayfun(@(i) estimate_derivative(z(i, :)), 1:size(z, 1), 'Uniform', 0);
  estimated_grad = mean(cat(3, grad_prediction_for_each_u{:}), 3);

  not_estimated_lambda = arrayfun(@(i) counts' * (matrices.Kru_grad(:, :, i)' -...
    S * matrices.Kuu_grad(:, :, i)) * Kuu_inv * gp.mu +...
    0.5 * (trace(matrices.Kuu_grad(:, :, i) * Kuu_inv * Sigma * Kuu_inv) +...
    gp.mu' * Kuu_inv * matrices.Kuu_grad(:, :, i) * Kuu_inv * gp.mu -...
    trace(matrices.Kuu_grad(:, :, i) * Kuu_inv)), 1:size(matrices.Kuu_grad, 3));
  not_estimated_mu = (counts' * S)' - 0.5 * (Kuu_inv + Kuu_inv') * gp.mu;
  not_estimated_elbo = counts' * S * gp.mu - 0.5 * (trace(Kuu_inv * Sigma) +...
    gp.mu' * Kuu_inv * gp.mu + log(det(matrices.Kuu)) - log(det(Sigma)));
  not_estimated_B = (Sigma_inv - Kuu_inv) * gp.B;

  not_estimated = vertcat(not_estimated_elbo, not_estimated_lambda',...
    diag(not_estimated_B), not_estimated_mu, get_lower_triangle(not_estimated_B));
  changes = not_estimated - 0.5 * estimated_grad;

  % TESTING
  %fprintf('Not estimated: %f, estimated: %f, total: %f\n', not_estimated(2), estimated_grad(2), changes(2));
  p_derivative = -0.5 * estimated_grad(2) + counts' * (matrices.Kru_grad(:, :, 1)' -...
    matrices.Kru' * Kuu_inv * matrices.Kuu_grad(:, :, 1)) * Kuu_inv * gp.mu;
  kl_derivative = 0.5 * (trace(Kuu_inv * matrices.Kuu_grad(:, :, 1) * Kuu_inv * Sigma) +...
    gp.mu' * Kuu_inv * matrices.Kuu_grad(:, :, 1) * Kuu_inv * gp.mu -...
    trace(Kuu_inv * matrices.Kuu_grad(:, :, 1)));
  fprintf('d/d_lambda0 of E[log p(D | r)]: %f\n', p_derivative);
  fprintf('average v: %f\n', 0.5 * estimated_grad(1));
  %fprintf('d/d_lambda0 of -KL(q(u) || p(u)): %f\n', kl_derivative);
  %fprintf('their sum: %f\n', changes(2));
  %assert(abs(p_derivative + kl_derivative - changes(2)) < 1e-5);

  elbo = changes(1);
  grad = transform_gradient(changes(2:end), vigpirlpackparam(gp), size(gp.lambda), size(gp.mu));

  %fprintf('Not estimated grad:');
  %disp(not_estimated(2));
  %fprintf('Estimated grad:');
  %disp(estimated_grad(2));
end

function a = extract(mdp, state, action)
  temp = mdp(state, action, :);
  a = reshape(temp, size(mdp, 3), 1);
end

function new_grad = transform_gradient(grad, hyperparameters, d, m)
  num_transformed = 1 + d + m;
  new_grad = vertcat(vigpirlhpxform(hyperparameters(1:num_transformed),...
    grad(1:num_transformed), 'exp', 2), grad(num_transformed+1:end));
end
