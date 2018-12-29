                      % Optimized kernel computation function for DC mode GPIRL.
function [K_uf, invK, K_ufKinv, K_ff, K_uu_grad, K_uf_grad] = vigpirlkernel(gp, Xstar)

                                % Constants.
  dims = length(gp.inv_widths);
  n = size(gp.X_u,1);

  % TODO: is this necessary?
                                                            % Undo transforms.
  inv_widths = gpirlhpxform(gp.inv_widths,[],gp.ard_xform,1); % This is \Lambda
  noise_var = gpirlhpxform(gp.noise_var,[],gp.noise_xform,1); % This is 2\sigma^2
  rbf_var = gpirlhpxform(gp.rbf_var,[],gp.rbf_xform,1); % This is \beta
  inv_widths = min(inv_widths,1e100); % Prevent overflow.

                                % Compute scales.
  iw_sqrt = sqrt(inv_widths);

                                % Scale positions in feature space.
  X_u_warped = gp.X_u;
  X_f_warped = gp.X;
  if nargin >= 2
    X_s_warped = Xstar;
  end
  X_u_scaled = bsxfun(@times,iw_sqrt,X_u_warped);
  X_f_scaled = bsxfun(@times,iw_sqrt,X_f_warped);

  function [nmat, nmat2, nconst, nconst2] = construct_noise_matrix(n)
    mask_mat = ones(n)-eye(n);
    nconst = exp(-0.5*noise_var*sum(inv_widths));
    nconst2 = -0.5 * noise_var;
    nmat = nconst*ones(n) + (1-nconst)*eye(n);
    nmat2 = nconst2*ones(n) + (1-nconst2)*eye(n);
  end

  function deriv = K_uu_lambda_i_derivatives(K_uu, X, Y, nmat2, i)
    xi = X(:, i);
    xi_sq = xi .^ 2;
    yi = Y(:, i);
    yi_sq = yi .^ 2;
    inner = repmat(xi_sq', size(Y, 1), 1) + repmat(yi_sq, 1, size(X, 1)) - 2 * yi * xi';
    deriv = K_uu .* (-0.5 * inner + nmat2);
  end

  function [K_uu, K_uu_grad, nconst] = compute_covariance_matrix(X)
    d_uu = bsxfun(@plus,sum(X.^2,2),sum(X.^2,2)') - 2 * X * X';
    d_uu = max(d_uu,0);
    [nmat, nmat2, nconst, nconst2] = construct_noise_matrix(size(X, 1));
    K_uu_deriv_lambda0 = exp(-0.5*d_uu).*nmat;
    K_uu = rbf_var * K_uu_deriv_lambda0;
    c = arrayfun(@(i) K_uu_lambda_i_derivatives(K_uu, X, X, nmat2, i), 1:size(X, 2), 'Uniform', 0);
    K_uu_grad = cat(3, K_uu_deriv_lambda0, c{:});
  end

  [K_ff, ~, ~] = compute_covariance_matrix(X_f_scaled);
  [K_uu, K_uu_grad, nconst] = compute_covariance_matrix(X_u_scaled);

  function [K_uf, K_uf_grad] = compute_uf_matrix(X_f_scaled)
    d_uf = bsxfun(@plus,sum(X_u_scaled.^2,2),sum(X_f_scaled.^2,2)') - 2*(X_u_scaled*(X_f_scaled'));
    d_uf = max(d_uf,0);
    K_uf_deriv_lambda0 = nconst * exp(-0.5*d_uf);
    K_uf = rbf_var * K_uf_deriv_lambda0;
    a = size(K_uf, 1);
    b = size(K_uf, 2);
    c = arrayfun(@(i) K_uu_lambda_i_derivatives(K_uf, X_f_scaled,...
      X_u_scaled, zeros(a, b), i), 1:size(X_f_scaled, 2), 'Uniform', 0);
    K_uf_grad = cat(3, K_uf_deriv_lambda0, c{:});
  end

  if nargin < 2
    [K_uf, K_uf_grad] = compute_uf_matrix(X_f_scaled);
  else
                                % Use Xstar to compute K_uf matrix.
    X_s_scaled = bsxfun(@times,iw_sqrt,X_s_warped);
    [K_uf, K_uf_grad] = compute_uf_matrix(X_s_scaled);
  end

                                % Invert the kernel matrix.
  try
    invK = vigpirlsafeinv(K_uu);
  catch err
                                % Display the error.
    rethrow(err);
  end
  K_ufKinv = K_uf'*invK;
end
