function tests = test_vigpirlkernel
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

  Kuu = [k(0, 0) k(1, 0); k(0, 1) k(1, 1)];
  Kuf = [k(0, 0) k(1, 0) k(2, 0); k(1, 0) k(1, 1) k(1, 2)];
  Kff = [k(0, 0) k(1, 0) k(2, 0); k(0, 1) k(1, 1) k(2, 1); k(0, 2) k(1, 2) k(2, 2)];
  Kuuinv = inv(Kuu);

  [K_uf, invK, K_ufKinv, K_ff] = vigpirlkernel(struct('X_u', Xu, 'X', X,...
    'noise_var', noise, 'rbf_var', rbf, 'inv_widths', [2 0]));
  verifyEqual(testCase, K_uf, Kuf, 'AbsTol', 1e-10);
  verifyEqual(testCase, invK, Kuuinv, 'AbsTol', 1e-10);
  verifyEqual(testCase, K_ufKinv, Kuf' * Kuuinv, 'AbsTol', 1e-10);
  verifyEqual(testCase, K_ff, Kff, 'AbsTol', 1e-10);
end