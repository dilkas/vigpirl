function changes = full_gradient(Sigma, mdp_model,...
  mdp_data, example_samples, counts, mu, matrices, z, T, B)

  Kuu_inv = inv(matrices.Kuu);
  Sigma_inv = inv(Sigma);
  r_covariance_matrix = matrices.Krr - matrices.Kru' * Kuu_inv * matrices.Kru;
  grad_prediction_for_each_u = arrayfun(@(i) estimate_derivative(z(i, :),...
    matrices, r_covariance_matrix, mdp_data, Sigma, mu, mdp_model,...
    example_samples, counts, T, B), 1:size(z, 1), 'Uniform', 0);
  estimated_grad = mean(cat(3, grad_prediction_for_each_u{:}), 3);

  not_estimated_lambda = arrayfun(@(i) counts' * (matrices.Kru_grad(:, :, i)' -...
    matrices.Kru' * Kuu_inv * matrices.Kuu_grad(:, :, i)) * Kuu_inv * mu +...
    0.5 * (trace(Kuu_inv * matrices.Kuu_grad(:, :, i) * Kuu_inv * Sigma) +...
    mu' * Kuu_inv * matrices.Kuu_grad(:, :, i) * Kuu_inv * mu -...
    trace(Kuu_inv * matrices.Kuu_grad(:, :, i))), 1:size(matrices.Kuu_grad, 3));
  not_estimated_mu = (counts' * matrices.Kru' * Kuu_inv)' - 0.5 * (Kuu_inv + Kuu_inv') * mu;
  not_estimated_elbo = counts' * matrices.Kru' * Kuu_inv * mu -...
    0.5 * (trace(Kuu_inv * Sigma) + mu' * Kuu_inv * mu + log(det(matrices.Kuu)) - log(det(Sigma)));
  not_estimated_B = (Sigma_inv - Kuu_inv) * B;

  not_estimated = vertcat(not_estimated_elbo, not_estimated_lambda',...
    diag(not_estimated_B), not_estimated_mu, get_lower_triangle(not_estimated_B));
  changes = not_estimated - 0.5 * estimated_grad;

  fprintf('Not estimated grad:');
  disp(not_estimated(2));
  fprintf('Estimated grad:');
  disp(estimated_grad(2));
end
