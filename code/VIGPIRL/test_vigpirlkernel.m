function tests = test_vigpirlkernel
  tests = functiontests(localfunctions);
end

function test_d_covariance_matrix_d_lambda_i(testCase)
  answer = d_covariance_matrix_d_lambda_i(ones(3, 2), [2 0; 4 0; 8 0], [1 0; 3 0], zeros(3, 2), 1);
  correct = -0.5 * [1 1; 9 1; 49 25];
  verifyEqual(testCase, answer, correct);
end
