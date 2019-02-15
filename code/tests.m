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

  %Xu = [0 0; 1 3];
  %X = [0 0; 1 0; 2 0];
  %noise = 0;
  %rbf = 0.1;
  %gp = struct('X_u', Xu, 'X', X, 'noise_var', noise, 'rbf_var', rbf,...
  %  'inv_widths', [2 0]);

  %Kuu = [k(0, 0) k(1, 0); k(0, 1) k(1, 1)];
  %Kuf = [k(0, 0) k(1, 0) k(2, 0); k(1, 0) k(1, 1) k(1, 2)];
  %Kff = [k(0, 0) k(1, 0) k(2, 0); k(0, 1) k(1, 1) k(2, 1); k(0, 2) k(1, 2) k(2, 2)];
  %Kuuinv = inv(Kuu);
  %KruKuu = Kuf' * Kuuinv;

  % Test vigpirlkernel
  %[K_uf, K_uu, invK, K_ufKinv, K_ff] = vigpirlkernel(gp);
  %verifyEqual(testCase, K_uf, Kuf, 'AbsTol', 1e-10);
  %verifyEqual(testCase, K_uu, Kuu, 'AbsTol', 1e-10);
  %verifyEqual(testCase, invK, Kuuinv, 'AbsTol', 1e-10);
  %verifyEqual(testCase, K_ufKinv, KruKuu, 'AbsTol', 1e-10);
  %verifyEqual(testCase, K_ff, Kff, 'AbsTol', 1e-10);

  % Test estimate_derivative
  demonstrations = {[1, 1], [3, 2]};
  counts = [1; 0; 1];
  u = [1 -2 3];
  mdp_data = struct('discount', 0, 'states', 3, 'actions', 2);
  mdp_data.sa_s(:, :, 1) = [2, 3; 1, 3; 1, 2];
  mdp_data.sa_p(1:3, 1:2, 1) = 1;
  gp = struct('mu', [-2; 3; 3], 'B', eye(3), 'lambda0', 1, 'lambda', 1);

  matrices = struct('Kuu', eye(3));
  matrices.Krr = [46 29 44; 29 36 27; 44 27 46];
  matrices.Kru = [-2 -5 -2; 5 3 4; -4 -1 -5];
  matrices.Kuu_grad(:, :, 1) = [2 -2 5; 5 3 -2; 5 3 1];
  matrices.Kuu_grad(:, :, 2) = [3 -5 -2; -5 -5 0; 2 4 -3];
  matrices.Kru_grad(:, :, 1) = [4 4 -4; -1 3 1; 5 3 0];
  matrices.Kru_grad(:, :, 2) = [0 5 3; 2 4 -4; -5 2 -4];
  matrices.Krr_grad(:, :, 1) = [-5 0 -2; 1 1 -5; -3 -3 5];
  matrices.Krr_grad(:, :, 2) = [-3 -5 1; 2 -1 2; -5 3 -4];

  addpaths;
  S = [-2 5 -4; -5 3 -1; -2 4 -5];
  r = S * u';
  mdp_solution = linearmdpsolve(mdp_data, r);
  v = counts' * mdp_solution.v;
  % mdp_values = [-23.3069; -13.3069; -24.3069];

  % Test full_gradient
  [~, full_gradient_answer] = full_gradient(mdp_data, demonstrations, counts, gp, u, matrices);
  full_gradient_correct_answer = 116 + 0.5 * (417 * v - 25);
  verifyEqual(testCase, full_gradient_answer(1), full_gradient_correct_answer);
end

function test_hyperparameter_packing(testCase)
  gp = struct('rbf_var', 0, 'inv_widths', [1 2],...
    'B', [3 0 0; 4 5 0; 6 7 8], 'mu', [-1; -2; -3]);
  verifyEqual(testCase, vigpirlunpackparam(gp, vigpirlpackparam(gp)), gp, 'AbsTol', 1e-10);
end
