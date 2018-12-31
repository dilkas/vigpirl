function changes = full_gradient(Sigma, Sigma_inv, mdp_model,...
  mdp_data, example_samples, counts, mu, Kru, Kuu, Kuu_inv, KruKuu, Krr,...
  Kuu_grad, Kru_grad, z)
  r_covariance_matrix = Krr - KruKuu * Kru;
  grad_prediction_for_each_u = arrayfun(@(i) estimate_derivative(z(i, :),...
    KruKuu, Kuu, Kuu_inv, r_covariance_matrix, Kuu_grad, mdp_data, Sigma, Sigma_inv,...
    mu, mdp_model, example_samples, counts), 1:size(z, 1), 'Uniform', 0);
  estimated_grad = mean(cat(3, grad_prediction_for_each_u{:}), 3);

  not_estimated_lambda = arrayfun(@(i) -0.5 *...
    trace(Kuu_inv * Kuu_grad(:, :, i)) + counts' * (Kru_grad(:, :, i)' -...
    KruKuu * Kuu_grad(:, :, i)) * Kuu_inv * mu, 1:size(Kuu_grad, 3));
  not_estimated_mu = counts' * Kru' * Kuu_inv;
  not_estimated_elbo = 0.5 * log(det(Sigma)) - 0.5 * log(det(Kuu)) +...
    counts' * KruKuu * mu;
  not_estimated = vertcat(not_estimated_elbo, not_estimated_lambda', not_estimated_mu');
  changes = not_estimated - 0.5 * estimated_grad;
end
