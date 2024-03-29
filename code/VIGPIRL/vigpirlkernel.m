                      % Optimized kernel computation function for DC mode GPIRL.
function matrices = vigpirlkernel(gp, Xstar)
  lambda = min(gp.lambda, 1e100); % Prevent overflow.
  iw_sqrt = sqrt(lambda); % Compute scales.

                                % Scale positions in feature space.
  X_u_warped = gp.X_u;
  X_f_warped = gp.X;
  if nargin >= 2
    X_s_warped = Xstar;
  end
  X_u_scaled = bsxfun(@times, iw_sqrt, X_u_warped);
  X_f_scaled = bsxfun(@times, iw_sqrt, X_f_warped);

  function [K_uu, K_uu_grad, nconst] = compute_covariance_matrix(X)
    d_uu = bsxfun(@plus, sum(X.^2, 2), sum(X.^2, 2)') - 2 * X * X';
    d_uu = max(d_uu, 0);
    [nmat, nmat2, nconst] = construct_noise_matrix(size(X, 1), gp.noise_var, lambda);
    K_uu_deriv_lambda0 = exp(-0.5*d_uu).*nmat;
    K_uu = gp.lambda0 * K_uu_deriv_lambda0;
    c = arrayfun(@(i) d_covariance_matrix_d_lambda_i(K_uu, X, X, nmat2, i),...
                 1:size(X, 2), 'Uniform', 0);
    K_uu_grad = cat(3, K_uu_deriv_lambda0, c{:});
  end

  [K_ff, K_ff_grad, ~] = compute_covariance_matrix(X_f_scaled);
  [K_uu, K_uu_grad, nconst] = compute_covariance_matrix(X_u_scaled);

  function [K_uf, K_uf_grad] = compute_uf_matrix(X_f_scaled)
    d_uf = bsxfun(@plus, sum(X_u_scaled .^ 2, 2), sum(X_f_scaled .^ 2, 2)') -...
      2 * X_u_scaled * X_f_scaled';
    d_uf = max(d_uf, 0);
    K_uf_deriv_lambda0 = nconst * exp(-0.5 * d_uf);
    K_uf = gp.lambda0 * K_uf_deriv_lambda0;
    a = size(K_uf, 1);
    b = size(K_uf, 2);
    c = arrayfun(@(i) d_covariance_matrix_d_lambda_i(K_uf, X_u_scaled,...
      X_f_scaled, zeros(a, b), i), 1:size(X_f_scaled, 2), 'Uniform', 0);
    K_uf_grad = cat(3, K_uf_deriv_lambda0, c{:});
  end

  if nargin < 2
    [K_uf, K_uf_grad] = compute_uf_matrix(X_f_scaled);
  else
                                % Use Xstar to compute K_uf matrix.
    X_s_scaled = bsxfun(@times, iw_sqrt, X_s_warped);
    [K_uf, K_uf_grad] = compute_uf_matrix(X_s_scaled);
  end

  matrices = struct('Kuu', K_uu, 'Kru', K_uf, 'Krr', K_ff, 'Kuu_grad',...
    K_uu_grad, 'Kru_grad', K_uf_grad, 'Krr_grad', K_ff_grad);
end

function [nmat, nmat2, nconst] = construct_noise_matrix(n, noise, lambda)
  nconst = exp(-0.5 * noise * sum(lambda));
  nconst2 = -0.5 * noise;
  nmat = nconst * ones(n) + (1 - nconst) * eye(n);
  nmat2 = nconst2 * ones(n) - nconst2 * eye(n);
end

function deriv = d_covariance_matrix_d_lambda_i(K, X, Y, nmat2, i)
  xi = X(:, i);
  xi_sq = xi .^ 2;
  yi = Y(:, i);
  yi_sq = yi .^ 2;
  inner = repmat(yi_sq', size(X, 1), 1) + repmat(xi_sq, 1, size(Y, 1))...
          - 2 * xi * yi';
  deriv = K .* (-0.5 * inner + nmat2);
end