                                % GP-based non-linear IRL algorithm.
function irl_result = vigpirlrun(algorithm_params,mdp_data,mdp_model,...
                                 feature_data,demonstrations,~,~)

                    % algorithm_params - parameters of the GP IRL algorithm.
                    % mdp_data - definition of the MDP to be solved.
                    % demonstrations - cell array containing examples.
                    % irl_result - result of IRL algorithm (see bottom of file).

                                % Fill in default parameters.
  algorithm_params = vigpirldefaultparams(algorithm_params);

                      % Get state-action counts and initial state distributions.
  [mu_sa, counts] = vigpirlgetstatistics(demonstrations, mdp_data);

                                % Create initial GP.
  gp = vigpirlinit(algorithm_params, feature_data);
                                % Choose inducing points.
  %gp = vigpirlgetinducingpoints(gp,mu_sa,algorithm_params);
  gp.X_u = gp.X; % TEMP
  m = size(gp.X_u, 1);
  gp.mu = rand(1, m)';
  %gp.mu = [-2; 3; 3]; % TEMP
                 % B is a lower triangular matrix with positive diagonal entries
  %gp.B = normrnd(0, 1, [m, m]);
  %gp.B(1:m+1:end) = random('Chisquare', 4, [m, 1]);
  %gp.B = tril(gp.B);
  gp.B = eye(m); % TEMP
  %gp.D = diag(normrnd(0, 1));
  gp.D = zeros(m, m); % TEMP

  d = size(feature_data.splittable, 2);
  elbo_list = [];
  grad_history = [];
  hyperparameter_history = [];

  % wrapper: vector of parameters -> scalar value * gradient vector
  function [elbo, grad] = wrapper(parameter_vector)
    gp = vigpirlunpackparam(gp, parameter_vector);
    %gp.lambda0 = exp(parameter_vector);

    %disp(gp);
    %disp(gp.B * gp.B');
    %disp(gp.mu);
 
    matrices = vigpirlkernel(gp);
    zz = mvnrnd(gp.mu', gp.B * gp.B', algorithm_params.samples_count);
    [elbo, grad] = full_gradient(mdp_data, demonstrations, counts, gp, zz, matrices);

    % Disable gradients that don't work
    grad(1) = 0; % lambda0
    grad(2:d+1) = 0; % lambda (except first)
    %grad(d+m+2:d+2*m+1) = 0; % mu
    grad(d+2:d+m+1) = 0; % B diagonal
    grad(d+2*m+2:end) = 0; % rest of B
    %disp(grad(1));

    elbo_list = horzcat(elbo_list, elbo);
    hyperparameter_history = horzcat(hyperparameter_history, parameter_vector);
    grad_history = horzcat(grad_history, grad);
    fprintf('elbo: %f\n', elbo);

    % We want to maximise the ELBO, while fmincon wants to minimize...
    grad = -grad;
    %grad = -grad(1);
    elbo = -elbo;
  end

  parameter_vector = vigpirlpackparam(gp);

  mu1_log = [];
  mu2_log = [];
  mu3_log = [];
  multiplier = 10;
  for mu1 = 1:10
    real_mu1 = multiplier * (mu1 - 5);
    parameter_vector(d+m+2) = real_mu1;
    for mu2 = 1:10
      real_mu2 = multiplier * (mu2 - 5);
      parameter_vector(d+m+3) = real_mu2;
      for mu3 = 1:10
        real_mu3 = multiplier * (mu3 - 5)
        parameter_vector(d+m+4) = real_mu3;
        wrapper(parameter_vector);
        mu1_log = [mu1_log real_mu1];
        mu2_log = [mu2_log real_mu2];
        mu3_log = [mu3_log real_mu3];
      end
    end
  end

  %max_negative = min(elbo_list);
  %if (max_negative >= 0)
  %  return;
  %end
  %max_positive = max(elbo_list);
  %if (max_positive <= 0)
  %  return;
  %end
  %[red, green, blue] = arrayfun(@(x) choose_color(x, max_negative, max_positive), elbo_list);

  % 2D plots
  for mu1 = 1:10
    indices = 100*(mu1-1)+1:100*mu1;
    other_mu = multiplier * ((1:10) - 5);
    subplot(5, 2, mu1);
    contourf(other_mu, other_mu, reshape(elbo_list(indices), [10, 10]), 30);
    xlabel('$\mu_2$', 'Interpreter', 'latex');
    ylabel('$\mu_3$', 'Interpreter', 'latex');
    t = ['$\mu_1 = ', num2str(multiplier * (mu1 - 5)), '$'];
    title(t, 'Interpreter', 'latex');
  end

  % 3D Plots
  %scatter3(mu1_log, mu2_log, mu3_log, 100, elbo_list, 'filled');
  %colorbar();
  %q = quiver3(mu1_log, mu2_log, mu3_log, grad_history(d+m+2, :), grad_history(d+m+3, :), grad_history(d+m+4, :));
  %q.LineWidth = 2;
  %quiver(mu2_log, mu3_log, grad_history(d+m+3, :), grad_history(d+m+4, :));
  %xlabel('$\mu_1$', 'Interpreter', 'latex');
  %ylabel('$\mu_2$', 'Interpreter', 'latex');
  %zlabel('$\mu_3$', 'Interpreter', 'latex');
  return;

  % Checking if the gradients are correct
  %options = optimoptions(@fminunc, 'SpecifyObjectiveGradient', true);
  %[optimal_lambda0, optimal_elbo, ~, output] = fminunc(@wrapper, parameter_vector, options);
  %disp(output);

  %stem(elbo_list);
  %plot_history(hyperparameter_history);
  %plot_history(grad_history);

  % Return corresponding reward function.
  %r = matrices.Kru' * inv(matrices.Kuu) * gp.mu;
  %solution = feval([mdp_model 'solve'], mdp_data, r);
  %v = solution.v;
  %q = solution.q;
  %p = solution.p;
  %disp(r);
  %disp(v);
  %disp(p);
  %irl_result = struct('r', r, 'v', v, 'p', p, 'q', q, 'model_itr', {{gp}},...
  %                    'r_itr', {{r}}, 'model_r_itr', {{r}}, 'p_itr', {{p}},...
  %                    'model_p_itr', {{p}}, 'time', 0, 'score', 0);
  %return;

  % for AdaGrad
  G = zeros(m + d + 1 + m*(m+1)/2, 1);

  % for AdaDelta
  %num_hyperparameters = m + d + 1 + m*(m+1)/2;
  %E_g = zeros(num_hyperparameters, 1);
  %E_x = zeros(num_hyperparameters, 1);
  %epsilon = 1e-6;
  %rho = 0.95;

  i = 0;
  tic;
  while true
    % Compute the gradient
    matrices = vigpirlkernel(gp);
    z = mvnrnd(gp.mu', gp.B * gp.B', algorithm_params.samples_count);
    [elbo, grad] = full_gradient(mdp_data, demonstrations, counts, gp, z, matrices);

    old_hyperparameters = vigpirlpackparam(gp);

    grad(1) = 0; % lambda0
    grad(2:d+1) = 0; % lambda (except first)
    %grad(d+m+2:d+2*m+1) = 0; % mu
    grad(d+m+3) = 0; % mu (second)
    grad(d+m+4) = 0; % mu (third)
    grad(d+2:d+m+1) = 0; % B diagonal
    grad(d+2*m+2:end) = 0; % rest of B

    %fprintf('Hyperparameters:\n');
    %disp(old_hyperparameters);
    %fprintf('Gradient:\n');
    %disp(grad);

    hyperparameter_history = horzcat(hyperparameter_history, old_hyperparameters);

    % Make the derivative of B weaker
    learning_rate_vector(1:length(grad), 1) = algorithm_params.learning_rate;
    learning_rate_vector(d+2:d+m+1) = algorithm_params.B_learning_rate;
    learning_rate_vector(d+2*m+2:end) = algorithm_params.B_learning_rate;

    % for AdaGrad
    G = G + grad .^ 2;
    rho = (algorithm_params.learning_rate / sqrt(G))';

    % for AdaDelta
    %E_g = rho * E_g + (1 - rho) * grad .^ 2;
    %delta = sqrt(E_x + epsilon) ./ sqrt(E_g + epsilon) .* grad;
    %E_x = rho * E_x + (1 - rho) * delta .^ 2;

    %hyperparameters = old_hyperparameters + delta;
    hyperparameters = old_hyperparameters + learning_rate_vector .* grad;
    %hyperparameters = old_hyperparameters + rho .* grad;
    gp = vigpirlunpackparam(gp, hyperparameters);

    disp(elbo);
    fprintf('----------\n');
    elbo_list = horzcat(elbo_list, elbo);
    grad_history = horzcat(grad_history, grad);

    if norm(hyperparameters - old_hyperparameters, 1) < algorithm_params.required_precision
      break;
    end

    i = i + 1;
    if (i >= algorithm_params.num_iterations)
      break;
    end
  end
  time = toc;

  stem(elbo_list);
  plot_history(hyperparameter_history);
  plot_history(grad_history);

  r = matrices.Kru' * inv(matrices.Kuu) * gp.mu;
  solution = feval([mdp_model 'solve'], mdp_data, r);
  v = solution.v;
  q = solution.q;
  p = solution.p;
  disp(hyperparameters);
  disp(r);
  disp(v);
  disp(p);
  irl_result = struct('r', r, 'v', v, 'p', p, 'q', q, 'model_itr', {{gp}},...
                      'r_itr', {{r}}, 'model_r_itr', {{r}}, 'p_itr', {{p}},...
                      'model_p_itr', {{p}}, 'time', 0, 'score', 0);
end

function plot_history(matrix)
  figure();
  %ylim([-400 100]);
  hold on;
  for row = 1:size(matrix, 1)
    plot(matrix(row,:));
  end
  hold off;
end

function [r, g, b] = choose_color(elbo_value, max_negative, max_positive)
  b = 0;
  if elbo_value < 0
    g = 0;
    r = elbo_value / max_negative;
  else
    r = 0;
    g = elbo_value / max_positive;
  end
end