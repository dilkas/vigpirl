% Compute the ELBO and its gradient
% NOTE: the gradient is for parameters in vector form
% - mdp_data, demonstrations are defined as usual (see Mdp.m)
% - counts: a vector of itegers, counting how many times each state was
%           occurred in the demonstrations
% - gp: the Gaussina process
% - u_samples: S*m matrix with S samples drawn from q(u)
% - matrices: various matrices computed by vigpirlkernel
% - deterministic_r: a Boolean variable used to set Gamma = 0 so that this
%                    function can be tested properly. In all other cases, leave
%                    this parameter unassigned.
function [elbo, grad] = compute_gradient(mdp_data, demonstrations, counts,...
  gp, u_samples, matrices, deterministic_r)

  % Use a point sample of u to estimate the part of the gradient under E[]
  function answer = estimate_derivative(u)

    % Return the derivative of the ELBO w.r.t. lambda_i.
    % (Actually, the part of the derivative inside E[] but without v)
    function answer = lambda_derivative(i)
      R = S * matrices.Kru_grad(:, :, i) - matrices.Krr_grad(:, :, i) +...
        (matrices.Kru_grad(:, :, i)' - S * matrices.Kuu_grad(:, :, i)) *...
        Kuu_inv * matrices.Kru;
      t = r' - S * u';
      answer = trace(R * adjoint(Gamma)) / det(Gamma) - t' * Gamma_inv * R *...
        Gamma_inv * t;
    end

    % The part of v specific to a single state-action pair (i.e., the
    % expresion inside the two outer summations)
    function answer = part_of_v_specific_to(state_action)
      state = state_action(1);
      action = state_action(2);
      next_states = take_third_dimension(mdp_data.sa_s, state, action);
      next_probabilities = take_third_dimension(mdp_data.sa_p, state, action);
      s = next_probabilities' * solution.v(next_states, 1);
      answer = solution.v(state, 1) - mdp_data.discount * s;
    end

    %if (nargin('compute_gradient') > 6)
    %  fprintf('Deterministic mode is on!\n');
    %  r = (S * u')';
    %else
    r = mvnrnd(S * u', Gamma);
    %end

    U = (u' - gp.mu) * (u' - gp.mu)';
    solution = linearmdpsolve(mdp_data, r');
    v_for_each_state_action = cellfun(@part_of_v_specific_to,...
      demonstrations, 'Uniform', 0);
    v = sum(cat(3, v_for_each_state_action{:}), 3);

    % Compute everything inside E[] except for v
    unique_lambda_part = arrayfun(@lambda_derivative,...
      1:size(matrices.Kuu_grad, 3));
    unique_mu_part = (u' - gp.mu)' * (Sigma_inv + Sigma_inv');
    unique_B_part = 2 * (Sigma_inv * U * Sigma_inv - Sigma_inv) * gp.B;

    unique_part = vertcat(2, unique_lambda_part', diag(unique_B_part),...
      unique_mu_part', get_lower_triangle(unique_B_part));
    answer = unique_part * v;
  end

  Sigma = gp.B * gp.B';
  Kuu_inv = inv(matrices.Kuu);

  % Quit if inverting Sigma fails
  warning('');
  Sigma_inv = inv(Sigma);
  [warning_message, ~] = lastwarn;
  if ~isempty(warning_message)
    elbo = 0;
    grad = [];
    return;
  end

  S = matrices.Kru' * Kuu_inv;
  % We need to round Gamma because otherwise it's not symmetric due to
  % numerical errors
  Gamma = round(matrices.Krr - S * matrices.Kru, 4);

  % Same behaviour as for Sigma
  Gamma_inv = inv(Gamma);
  [warning_message, ~] = lastwarn;
  if ~isempty(warning_message)
    grad = [];
    elbo = 0;
    return;
  end

  % Calculate the estimated part
  grad_estimate_for_each_u = arrayfun(@(i)...
    estimate_derivative(u_samples(i, :)), 1:size(u_samples, 1), 'Uniform', 0);
  estimated_grad = mean(cat(3, grad_estimate_for_each_u{:}), 3);

  % Calculate the exact part
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

  % Gradient (and the ELBO) before the transformations
  changes = not_estimated - 0.5 * estimated_grad;

  elbo = changes(1);
  grad = transform_gradient(changes(2:end), vigpirlpackparam(gp),...
    size(gp.lambda), size(gp.mu));
end

% Used to select next states and their probabilities for a given state-action pair
function a = take_third_dimension(transition_matrix, state, action)
  temp = transition_matrix(state, action, :);
  a = reshape(temp, size(transition_matrix, 3), 1);
end

% Transform the gradient w.r.t. parameters into the same gradient w.r.t. the
% same parameters in vector form. For exponential transformations, this means
% multiplying by exp(parameter)
% - d: number of features
% - m: number of inducing points
function new_grad = transform_gradient(grad, parameters, d, m)
  % The vector is set up so that all parameters that require transformations
  % are at the start of the vector
  num_transformed = 1 + d + m;
  new_grad = vertcat(vigpirlhpxform(parameters(1:num_transformed),...
    grad(1:num_transformed), 'exp', 2), grad(num_transformed+1:end));
end
