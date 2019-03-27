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
  gp.X_u = gp.X;
  m = size(gp.X_u, 1);
  gp.mu = rand(1, m)';
  %gp.mu = [-5; 0; 4]; % TEMP

  % B is a lower triangular matrix with positive diagonal entries
  %gp.B = normrnd(0, 1, [m, m]);
  %gp.B(1:m+1:end) = random('Chisquare', 4, [m, 1]);
  %gp.B = tril(gp.B);
  gp.B = eye(m); % TEMP

  d = size(feature_data.splittable, 2);
  elbo_list = [];
  grad_history = [];
  hyperparameter_history = [];
  policy_history = [];

  % wrapper: vector of parameters -> scalar value * gradient vector
  function [elbo, grad] = wrapper(parameter_vector)
    gp = vigpirlunpackparam(gp, parameter_vector);
    matrices = vigpirlkernel(gp);
    zz = mvnrnd(gp.mu', gp.B * gp.B', algorithm_params.samples_count);
    [elbo, grad] = full_gradient(mdp_data, demonstrations, counts, gp, zz, matrices);

    % Disable gradients that don't work
    %grad(1) = 0; % lambda0
    %grad(2:d+1) = 0; % lambda (except first)
    %grad(d+m+2:d+2*m+1) = 0; % mu
    %grad(d+2:d+m+1) = 0; % B diagonal
    %grad(d+2*m+2:end) = 0; % rest of B

    elbo_list = horzcat(elbo_list, elbo);
    hyperparameter_history = horzcat(hyperparameter_history, parameter_vector);
    grad_history = horzcat(grad_history, grad);
    %fprintf('elbo: %f\n', elbo);

    % We want to maximise the ELBO, while fmincon wants to minimize...
    grad = -grad;
    elbo = -elbo;
  end

  parameter_vector = vigpirlpackparam(gp);

  % ELBO as a function of mu over 10 plots
  %mu1_log = [];
  %mu2_log = [];
  %mu3_log = [];
  %multiplier = 10;
  %for mu1 = 1:10
  %  real_mu1 = multiplier * (mu1 - 5);
  %  parameter_vector(d+m+2) = real_mu1;
  %  for mu2 = 1:10
  %    real_mu2 = multiplier * (mu2 - 5);
  %    parameter_vector(d+m+3) = real_mu2;
  %    for mu3 = 1:10
  %      real_mu3 = multiplier * (mu3 - 5)
  %      parameter_vector(d+m+4) = real_mu3;
  %      wrapper(parameter_vector);
  %      mu1_log = [mu1_log real_mu1];
  %      mu2_log = [mu2_log real_mu2];
  %      mu3_log = [mu3_log real_mu3];
  %    end
  %  end
  %end
  %for mu1 = 1:10 indices = 100*(mu1-1)+1:100*mu1;
  %     other_mu = multiplier * ((1:10) - 5);
  %  subplot(5, 2, mu1);
  %  contourf(other_mu, other_mu, reshape(elbo_list(indices), [10, 10]), 30);
  %  xlabel('$\mu_2$', 'Interpreter', 'latex');
  %  ylabel('$\mu_3$', 'Interpreter', 'latex');
  %  t = ['$\mu_1 = ', num2str(multiplier * (mu1 - 5)), '$'];
  %  title(t, 'Interpreter', 'latex');
  %end
  %return;

  % Policy as a function of rewards over 10 plots
  %mu1_log = [];
  %mu2_log = [];
  %mu3_log = [];
  %policy_log = [];
  %mu1_range = -5:5;
  %mu_range = -5:0.5:5;
  %num_points = length(mu_range) * length(mu_range);
  %for mu1 = mu1_range
  %  for mu2 = mu_range
  %    for mu3 = mu_range
  %      solution = feval([mdp_model 'solve'], mdp_data, [mu1; mu2; mu3]);
  %      policy = solution.p(1, 1); % state 1, action 1
  %      policy_log = [policy_log policy];
  %      mu1_log = [mu1_log mu1];
  %      mu2_log = [mu2_log mu2];
  %      mu3_log = [mu3_log mu3];
  %    end
  %  end
  %end
  %for mu1 = 1:length(mu1_range)
  %  indices = num_points * (mu1-1) + (1:num_points);
  %  subplot(5, 2, mu1);
  %  contourf(mu_range, mu_range, reshape(policy_log(indices), [length(mu_range), length(mu_range)]), 30);
  %  xlabel('$r(s_2)$', 'Interpreter', 'latex');
  %  ylabel('$r(s_3)$', 'Interpreter', 'latex');
  %  t = ['$r(s_1) = ', num2str(mu1 - 5), '$'];
  %  title(t, 'Interpreter', 'latex');
  %end
  %return;

  % ELBO as a function of mu for various values of gamma
  %mu_from = -100;
  %mu_to = 50;
  %gamma_from = 0;
  %gamma_to = 9;
  %colours = colormap('jet');
  %indices = round(linspace(1, length(colours), gamma_to - gamma_from + 1));
  %set(groot, 'defaultAxesColorOrder', colours(indices, :));
  %figure('Units', 'centimeters', 'Position', [0 0 15 10],...
  %       'PaperPositionMode', 'auto');
  %for gamma = gamma_from:gamma_to
  %  mdp_data.discount = gamma / 10.0;
  %  mu_log = [];
  %  for mu = mu_from:mu_to
  %    parameter_vector(d+m+2:d+m+4) = mu;
  %    wrapper(parameter_vector);
  %    mu_log = [mu_log mu];
  %  end
  %  plot(mu_log, elbo_list((length(elbo_list)-(mu_to - mu_from)):end));
  %  if (gamma == 0)
  %    hold on;
  %  end
  %end
  %legend('$\gamma = ' + string((gamma_from:gamma_to) / 10.0) + '$', 'Interpreter', 'latex');
  %xlabel('$\mu_1 = \mu_2 = \mu_3$', 'Interpreter', 'latex');
  %ylabel('$\mathcal{L}$', 'Interpreter', 'latex');
  %hold off;
  %print('../mpaper/elbo_over_gamma', '-depsc2');

  % ELBO and derivative over some element of B
  % NOTE: this is the log of the actual value
  % TODO: titles, fix values on the x-axis, add two axes and their names to the y-aixs
  %accuracy = 0.1;
  %mu_from = -3;
  %mu_to = 3;
  %for i = 1:3
  %  mu_log = [];
  %  index = d + 1 + i;
  %  for mu = mu_from:accuracy:mu_to
  %    parameter_vector(index) = mu;
  %    wrapper(parameter_vector);
  %    mu_log = [mu_log mu];
  %  end
  %  parameter_vector(index) = 0;
  %  first_index = size(grad_history, 2) - length(mu_log) + 1;
  %  subplot(2, 3, i);
  %  yyaxis left;
  %  plot(mu_log, elbo_list(first_index:end));
  %  hold on;
  %  yyaxis right;
  %  plot(mu_log, grad_history(index, first_index:end));
  %  hold off;
  %end
  %mu_from = -5;
  %mu_to = 5;
  %for i = 1:3
  %  mu_log = [];
  %  index = d + 2 * m + 1 + i;
  %  for mu = mu_from:accuracy:mu_to
  %    parameter_vector(index) = mu;
  %    wrapper(parameter_vector);
  %    mu_log = [mu_log mu];
  %  end
  %  parameter_vector(index) = 0;
  %  first_index = size(grad_history, 2) - length(mu_log) + 1;
  %  subplot(2, 3, i + 3);
  %  yyaxis left;
  %  plot(mu_log, elbo_list(first_index:end));
  %  hold on;
  %  yyaxis right;
  %  plot(mu_log, grad_history(index,first_index:end));
  %  hold off;
  %end
  %return;

  % ELBO and derivative over some element of B (more selective and polished version)
  %figure('Units', 'centimeters', 'Position', [0 0 15 20], 'PaperPositionMode', 'auto');
  %subplot(2, 1, 1);
  %accuracy = 0.05;
  %B_from = 1;
  %B_to = 10;
  %B_log = [];
  %index = d + 2;
  %for B = B_from:accuracy:B_to
  %  parameter_vector(index) = log(B);
  %  wrapper(parameter_vector);
  %  B_log = [B_log B];
  %end
  %parameter_vector(index) = 0;
  %first_index = size(grad_history, 2) - length(B_log) + 1;
  %xlabel('$B_{1,1}$', 'Interpreter', 'latex');
  %yyaxis left;
  %plot(B_log, elbo_list(first_index:end));
  %ylabel('$E[v]$', 'Interpreter', 'latex');
  %yyaxis right;
  %plot(B_log, grad_history(index, first_index:end));
  %ylabel('$\partial E[v]/\partial B_{1,1}$', 'Interpreter', 'latex');

  %subplot(2, 1, 2);
  %B_from = -100;
  %B_to = 100;
  %accuracy = 1;
  %B_log = [];
  %index = d + 2 * m + 2;
  %for B = B_from:accuracy:B_to
  %  parameter_vector(index) = B;
  %  wrapper(parameter_vector);
  %  B_log = [B_log B];
  %end
  %parameter_vector(index) = 0;
  %first_index = size(grad_history, 2) - length(B_log) + 1;
  %xlabel('$B_{2,1}$', 'Interpreter', 'latex');
  %yyaxis left;
  %plot(B_log, elbo_list(first_index:end));
  %ylabel('$E[v]$', 'Interpreter', 'latex');
  %yyaxis right;
  %plot(B_log, grad_history(index, first_index:end));
  %ylabel('$\partial E[v]/\partial B_{2,1}$', 'Interpreter', 'latex');

  %print('../mpaper/elbo_over_B_short', '-depsc2');
  %return;

  % Checking if the gradients are correct
  %options = optimoptions(@fminunc, 'SpecifyObjectiveGradient', true, 'CheckGradients', true);
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
  num_hyperparameters = m + d + 1 + m*(m+1)/2;
  E_g = zeros(num_hyperparameters, 1);
  E_x = zeros(num_hyperparameters, 1);
  epsilon = 1e-6;
  rho = 0.6;

  i = 0;
  tic;
  while true
    fprintf('.');
    % Compute the gradient
    matrices = vigpirlkernel(gp);
    z = mvnrnd(gp.mu', gp.B * gp.B', algorithm_params.samples_count);
    [elbo, grad] = full_gradient(mdp_data, demonstrations, counts, gp, z, matrices);
    if (isempty(grad))
      break;
    end

    old_hyperparameters = vigpirlpackparam(gp);

    %grad(1) = 0; % lambda0
    %grad(2:d+1) = 0; % lambda (except first)
    %grad(d+m+2:d+2*m+1) = 0; % mu
    %grad(d+2:d+m+1) = 0; % B diagonal
    %grad(d+2*m+2:end) = 0; % rest of B

    hyperparameter_history = horzcat(hyperparameter_history, old_hyperparameters);

    % Make the derivative of B weaker
    learning_rate_vector(1:length(grad), 1) = algorithm_params.learning_rate;
    learning_rate_vector(d+2:d+m+1) = algorithm_params.B_learning_rate;
    learning_rate_vector(d+2*m+2:end) = algorithm_params.B_learning_rate;
    learning_rate_vector(2) = algorithm_params.lambda1_learning_rate;

    % for AdaGrad
    %G = G + grad .^ 2;
    %rho = (algorithm_params.learning_rate / sqrt(G))';

    % for AdaDelta
    %E_g = rho * E_g + (1 - rho) * grad .^ 2;
    %delta = sqrt(E_x + epsilon) ./ sqrt(E_g + epsilon) .* grad;
    %E_x = rho * E_x + (1 - rho) * delta .^ 2;

    %hyperparameters = old_hyperparameters + delta;
    hyperparameters = old_hyperparameters + learning_rate_vector .* grad;
    %hyperparameters = old_hyperparameters + rho .* grad;
    gp = vigpirlunpackparam(gp, hyperparameters);

    %fprintf('ELBO: %f\n', elbo);
    %fprintf('----------\n');
    elbo_list = horzcat(elbo_list, elbo);
    grad_history = horzcat(grad_history, grad);

    r = matrices.Kru' * inv(matrices.Kuu) * gp.mu;
    solution = feval([mdp_model 'solve'], mdp_data, r);
    policy_history = horzcat(policy_history, [solution.p(1, 1); solution.p(2, 1); solution.p(3, 2)]);

    if norm(hyperparameters - old_hyperparameters, 1) < algorithm_params.required_precision
      break;
    end

    i = i + 1;
    if (i >= algorithm_params.num_iterations)
      break;
    end
  end
  fprintf('\n');
  time = toc;

  figure('Units', 'centimeters', 'Position', [0 0 12 16], 'PaperPositionMode', 'auto');
  subplot(2, 1, 1);
  stem(elbo_list);
  xlabel('number of iterations');
  ylabel('$\mathcal{L}$', 'Interpreter', 'latex');

  subplot(2, 1, 2);
  plot_history(policy_history);
  xlabel('number of iterations');
  legend('$\pi(a_1 \mid s_1)$', '$\pi(a_1 \mid s_2)$', '$\pi(a_2 \mid s_3)$', 'Interpreter', 'latex', 'Location', 'east');
  %print('../mpaper/figures/convergence_new', '-depsc2');

  figure('Units', 'centimeters', 'Position', [0 0 15 5], 'PaperPositionMode', 'auto');
  plot(hyperparameter_history(1,:), 'k-');
  hold on;
  plot(hyperparameter_history(2,:), 'k--');
  plot(hyperparameter_history(6,:), 'b');
  plot(hyperparameter_history(7,:), 'b--');
  plot(hyperparameter_history(8,:), 'b-.');
  plot(exp(hyperparameter_history(3,:)), 'r');
  plot(exp(hyperparameter_history(4,:)), 'r--');
  plot(exp(hyperparameter_history(5,:)), 'r-.');
  plot(hyperparameter_history(9,:), 'g');
  plot(hyperparameter_history(10,:), 'g--');
  plot(hyperparameter_history(11,:), 'g-.');
  legend('$\log \lambda_0$', '$\log \lambda_1$', '$\mu_1$', '$\mu_2$', '$\mu_3$', '$B_{1,1}$', '$B_{2,2}$', '$B_{3,3}$', '$B_{2,1}$', '$B_{3,1}$', '$B_{3,2}$', 'Interpreter', 'latex', 'Location', 'westoutside');
  %plot(exp(hyperparameter_history(1,:)), 'k-');
  %hold on;
  %plot(hyperparameter_history(2,:), 'k--');
  %plot(hyperparameter_history(6,:), 'b');
  %plot(hyperparameter_history(7,:), 'b--');
  %plot(hyperparameter_history(8,:), 'b-.');
  %legend('$\lambda_0$', '$\log\lambda_1$', '$\mu_1$', '$\mu_2$', '$\mu_3$', 'Interpreter', 'latex', 'Location', 'westoutside');
  xlabel('number of iterations');
  hold off;
  %print('../mpaper/figures/parameter_convergence_new', '-depsc2');

  matrices = vigpirlkernel(gp);
  r = matrices.Kru' * inv(matrices.Kuu) * gp.mu;
  solution = feval([mdp_model 'solve'], mdp_data, r);
  v = solution.v;
  q = solution.q;
  p = solution.p;
  %fprintf('Parameters:\n');
  %disp(hyperparameters);
  %fprintf('Rewards:\n');
  %disp(r);
  %fprintf('values:\n');
  %disp(v);
  %fprintf('Policy:\n');
  %disp(p);
  irl_result = struct('r', r, 'v', v, 'p', p, 'q', q, 'model_itr', {{gp}},...
                      'r_itr', {{r}}, 'model_r_itr', {{r}}, 'p_itr', {{p}},...
                      'model_p_itr', {{p}}, 'time', 0, 'score', 0, 'matrices', matrices, 'gp', gp);
end

function plot_history(matrix)
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