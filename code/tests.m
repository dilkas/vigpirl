function tests = tests
  addpaths;
  tests = functiontests(localfunctions);
end

function test_gradient(testCase)
  function c = k(x, y)
    c = rbf * exp(-0.5 * 2 * (x - y)^2);
  end

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

  S = [-2 5 -4; -5 3 -1; -2 4 -5];
  r = S * u';
  mdp_solution = linearmdpsolve(mdp_data, r);
  v = counts' * mdp_solution.v;

  [~, compute_gradient_answer] = compute_gradient(mdp_data, demonstrations, counts, gp, u, matrices, true);

  lambda0_derivative = 116 + 0.5 * (417 * v - 25);
  verifyEqual(testCase, compute_gradient_answer(1), lambda0_derivative);

  B11_derivative = -8 * v;
  B22_derivative = -24 * v;
  B33_derivative = v;
  B21_derivative = 15 * v;
  B31_derivative = 0;
  B32_derivative = 0;

  verifyEqual(testCase, compute_gradient_answer(3), B11_derivative);
  verifyEqual(testCase, compute_gradient_answer(4), B22_derivative);
  verifyEqual(testCase, compute_gradient_answer(5), B33_derivative);

  verifyEqual(testCase, compute_gradient_answer(9), B21_derivative);
  verifyEqual(testCase, compute_gradient_answer(10), B31_derivative);
  verifyEqual(testCase, compute_gradient_answer(11), B32_derivative);
end

function test_hyperparameter_packing(testCase)
  gp = struct('lambda0', 0, 'lambda', [1 2],...
    'B', [3 0 0; 4 5 0; 6 7 8], 'mu', [-1; -2; -3]);
  verifyEqual(testCase, vigpirlunpackparam(gp, vigpirlpackparam(gp)), gp, 'AbsTol', 1e-10);
end
