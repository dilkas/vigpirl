function tests = tests
  tests = functiontests(localfunctions);
end

function test_d_covariance_matrix_d_lambda_i(testCase)
  answer = d_covariance_matrix_d_lambda_i(ones(3, 2), [2 0; 4 0; 8 0],...
    [1 0; 3 0], zeros(3, 2), 1);
  correct = -0.5 * [1 1; 9 1; 49 25];
  verifyEqual(testCase, answer, correct);
end

function test_remaining_vigpirlkernel(testCase)
  function c = k(x, y)
    c = rbf * exp(-0.5 * 2 * (x - y)^2);
  end

  Xu = [0 0; 1 3];
  X = [0 0; 1 0; 2 0];
  noise = 0;
  rbf = 0.1;
  gp = struct('X_u', Xu, 'X', X, 'noise_var', noise, 'rbf_var', rbf,...
    'inv_widths', [2 0]);

  Kuu = [k(0, 0) k(1, 0); k(0, 1) k(1, 1)];
  Kuf = [k(0, 0) k(1, 0) k(2, 0); k(1, 0) k(1, 1) k(1, 2)];
  Kff = [k(0, 0) k(1, 0) k(2, 0); k(0, 1) k(1, 1) k(2, 1); k(0, 2) k(1, 2) k(2, 2)];
  Kuuinv = inv(Kuu);
  KruKuu = Kuf' * Kuuinv;

  % Test vigpirlkernel
  [K_uf, invK, K_ufKinv, K_ff] = vigpirlkernel(gp);
  verifyEqual(testCase, K_uf, Kuf, 'AbsTol', 1e-10);
  verifyEqual(testCase, invK, Kuuinv, 'AbsTol', 1e-10);
  verifyEqual(testCase, K_ufKinv, KruKuu, 'AbsTol', 1e-10);
  verifyEqual(testCase, K_ff, Kff, 'AbsTol', 1e-10);

  % Test estimate_derivative
  u = [1 -2];
  mdp = struct('sa_s', [2; 3; 1], 'sa_p', [1; 1; 1], 'discount', 0,...
    'states', 3, 'actions', 1);
  Kuuinv = eye(2);
  KruKuu = [1 2; 3 4; 5 6];
  Kuu_grad(:, :, 1) = [-1 0; 1 -1];
  Kuu_grad(:, :, 2) = [0 1; -1 0];
  r_covariance_matrix = zeros(3);
  Sigma_inv = eye(2);
  mu = [0 0];
  demonstrations = {[1, 1]};
  mdp_model = 'linearmdp';
  estimated_grad = [15; 0; -6; 12];

  answer = estimate_derivative(u, KruKuu, Kuuinv, r_covariance_matrix,...
    Kuu_grad, mdp, Sigma_inv, mu, mdp_model, demonstrations);
  verifyEqual(testCase, answer, estimated_grad);

  % Test full_gradient
  Sigma = eye(2);
  mu = [-1 1];
  counts = [1; 0; 0];
  Kru = [1 2 3; 4 5 6];
  KruKuu = Kru' * Kuuinv;
  Kru_grad(:, :, 1) = [-2 -1 0; 1 2 3];
  Kru_grad(:, :, 2) = [-1 0 1; 2 3 4];
  %Kru_grad(:, :, 3) = [0 1 2; 3 4 5];
  %Kru_grad(:, :, 4) = [-3 -2 -1; 0 1 2];
  %Kuu_grad(:, :, 3) = [1 -1; 0 1];
  %Kuu_grad(:, :, 4) = [-1 0; 1 -1];
                         % turns the covariance matrix of r into the zero matrix
  Krr = [17 22 27; 22 29 36; 27 36 45];
  z = [1 -2];
  grad = full_gradient(Sigma, mdp_model, mdp, demonstrations, counts, mu,...
                       Kru, Kuuinv, KruKuu, Krr, Kuu_grad, Kru_grad, z);
  verifyEqual(testCase, grad, [11; -2; 1; 4] - 0.5 * [35; 0; -28; 42]);
end
