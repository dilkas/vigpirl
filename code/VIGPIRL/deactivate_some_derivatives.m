function gradient = deactivate_some_derivatives(gradient, algorithm_params, d, m)
  if algorithm_params.disable_optimising_lambda0
    gradient(1) = 0;
  end
  if algorithm_params.disable_optimising_lambda
    gradient(2:d+1) = 0;
  end
  if algorithm_params.disable_optimising_mu
    grad(d+m+2:d+2*m+1) = 0;
  end
  if algorithm_params.disable_optimising_B_diagonal
    grad(d+2:d+m+1) = 0;
  end
  if algorithm_params.disable_optimising_rest_of_B
    grad(d+2*m+2:end) = 0;
  end
end