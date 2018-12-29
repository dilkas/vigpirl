function deriv = d_covariance_matrix_d_lambda_i(K, X, Y, nmat2, i)
  xi = X(:, i);
  xi_sq = xi .^ 2;
  yi = Y(:, i);
  yi_sq = yi .^ 2;
  inner = repmat(yi_sq', size(X, 1), 1) + repmat(xi_sq, 1, size(Y, 1))...
          - 2 * xi * yi';
  deriv = K .* (-0.5 * inner + nmat2);
end
