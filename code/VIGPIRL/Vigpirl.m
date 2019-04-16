% This class contains methods for running the algorithm as well as many
% experiments and plots. Some plotting methods run their own experiments, some
% should be called after run().
classdef Vigpirl
    properties
        algorithm_params
        counts              % Number of times each state appears in the demonstrations
        d                   % Number of features per state
        demonstrations
        gp
        m                   % Number of inducing points
        mdp_data

        % Properties used to track how things change with each iteration
        elbo_history = [];
        parameter_history = [];
        grad_history = [];
        policy_history = [];
    end
    methods
        % mdp should belong to class Mdp, algorithm_params can be an empty struct
        function obj = Vigpirl(mdp, algorithm_params)
            % Fill in default parameters
            obj.algorithm_params = vigpirldefaultparams(algorithm_params);
            % Get state-action counts and initial state distributions
            [mu_sa, obj.counts] = vigpirlgetstatistics(mdp.demonstrations, mdp.mdp_data);
            % Create initial GP
            obj.gp = vigpirlinit(obj.algorithm_params, mdp.feature_data);

            % Choose inducing points.
            %obj.gp = vigpirlgetinducingpoints(obj.gp, mu_sa, obj.algorithm_params);
            obj.gp.X_u = obj.gp.X;

            obj.d = size(mdp.feature_data.splittable, 2);
            obj.m = size(obj.gp.X_u, 1);
            obj.gp.mu = rand(1, obj.m)';
            obj.mdp_data = mdp.mdp_data;
            obj.demonstrations = mdp.demonstrations;

            % B is a lower triangular matrix with positive diagonal entries
            if obj.algorithm_params.random_initial_B
                obj.gp.B = normrnd(0, 1, [m, m]);
                obj.gp.B(1:m+1:end) = random('Chisquare', 4, [obj.m, 1]);
                obj.gp.B = tril(obj.gp.B);
            else
                obj.gp.B = eye(obj.m);
            end
        end

        % Run the optimisation algorithm. Updates the object with new values of
        % gp, extends all the history properties, and returns running time
        function [obj, time] = run(obj)
            % Set up variables for dynamic step size algorithms
            num_parameters = obj.m + obj.d + 1 + obj.m * (obj.m + 1) / 2;
            if strcmp(obj.algorithm_params.step_size_algorithm, 'AdaGrad')
                G = zeros(num_parameters, 1);
            elseif strcmp(obj.algorithm_params.step_size_algorithm, 'AdaDelta')
                E_g = zeros(num_parameters, 1);
                E_x = zeros(num_parameters, 1);
                epsilon = 1e-6;
                rho = 0.6;
            end

            tic;
            for i = 1:obj.algorithm_params.num_iterations
                fprintf('.');
                % Compute the gradient
                matrices = vigpirlkernel(obj.gp);
                u_samples = mvnrnd(obj.gp.mu', obj.gp.B * obj.gp.B',...
                    obj.algorithm_params.samples_count);
                [elbo, grad] = compute_gradient(obj.mdp_data,...
                    obj.demonstrations, obj.counts, obj.gp, u_samples, matrices);

                % If computing the gradient failed, fail here as well
                if (isempty(grad))
                    break;
                end

                old_parameters = vigpirlpackparam(obj.gp);
                grad = deactivate_some_derivatives(grad,...
                    obj.algorithm_params, obj.d, obj.m);

                % Update parameter values
                switch obj.algorithm_params.step_size_algorithm
                case 'AdaGrad'
                    G = G + grad .^ 2;
                    rho = (obj.algorithm_params.learning_rate / sqrt(G))';
                    parameters = old_parameters + rho .* grad;
                case 'AdaDelta'
                    E_g = rho * E_g + (1 - rho) * grad .^ 2;
                    delta = sqrt(E_x + epsilon) ./ sqrt(E_g + epsilon) .* grad;
                    E_x = rho * E_x + (1 - rho) * delta .^ 2;
                    rparameters = old_parameters + delta;
                otherwise
                    % Construct a step size vector according to the given parameters
                    step_size_vector(1:length(grad), 1) =...
                        obj.algorithm_params.learning_rate;
                    step_size_vector(obj.d + 2:obj.d + obj.m + 1) =...
                        obj.algorithm_params.B_learning_rate;
                    step_size_vector(obj.d + 2 * obj.m + 2:end) =...
                        obj.algorithm_params.B_learning_rate;
                    step_size_vector(2) =...
                        obj.algorithm_params.lambda1_learning_rate;
                    parameters = old_parameters + step_size_vector .* grad;
                end

                %fprintf('ELBO: %f\n', elbo);
                obj.gp = vigpirlunpackparam(obj.gp, parameters);

                % Update history properties
                obj.parameter_history = horzcat(obj.parameter_history, old_parameters);
                obj.elbo_history = horzcat(obj.elbo_history, elbo);
                obj.grad_history = horzcat(obj.grad_history, grad);
                r = matrices.Kru' * inv(matrices.Kuu) * obj.gp.mu;
                solution = linearmdpsolve(obj.mdp_data, r);
                obj.policy_history = horzcat(obj.policy_history,...
                    [solution.p(1, 1); solution.p(2, 1); solution.p(3, 2)]);

                if norm(parameters - old_parameters, 1) <...
                    obj.algorithm_params.required_precision
                    break;
                end
            end
            time = toc;
            fprintf('\n');
        end

        % Mostly for debugging
        function convergence_plots(obj)
            %stem(obj.elbo_history);
            plot_history(obj.parameter_history);
            figure();
            plot_history(obj.grad_history);
        end

        function elbo_and_policy_convergence_plots(obj)
            figure('Units', 'centimeters', 'Position', [0 0 12 16],...
                'PaperPositionMode', 'auto');
            subplot(2, 1, 1);
            stem(obj.elbo_history);
            xlabel('number of iterations');
            ylabel('$\mathcal{L}$', 'Interpreter', 'latex');
            subplot(2, 1, 2);
            plot_history(obj.policy_history);
            xlabel('number of iterations');
            legend('$\pi(a_1 \mid s_1)$', '$\pi(a_1 \mid s_2)$',...
                '$\pi(a_2 \mid s_3)$', 'Interpreter', 'latex', 'Location', 'east');
            %print('../mpaper/figures/convergence_new', '-depsc2');
        end

        function parameter_convergence_plot(obj)
            figure('Units', 'centimeters', 'Position', [0 0 15 5],...
                'PaperPositionMode', 'auto');
            plot(obj.parameter_history(1,:), 'k-');
            hold on;
            plot(obj.parameter_history(2,:), 'k--');
            plot(obj.parameter_history(6,:), 'b');
            plot(obj.parameter_history(7,:), 'b--');
            plot(obj.parameter_history(8,:), 'b-.');
            plot(exp(obj.parameter_history(3,:)), 'r');
            plot(exp(obj.parameter_history(4,:)), 'r--');
            plot(exp(obj.parameter_history(5,:)), 'r-.');
            plot(obj.parameter_history(9,:), 'g');
            plot(obj.parameter_history(10,:), 'g--');
            plot(obj.parameter_history(11,:), 'g-.');
            legend('$\log \lambda_0$', '$\log \lambda_1$', '$\mu_1$',...
                '$\mu_2$', '$\mu_3$', '$B_{1,1}$', '$B_{2,2}$', '$B_{3,3}$',...
                '$B_{2,1}$', '$B_{3,1}$', '$B_{3,2}$',...
                'Interpreter', 'latex', 'Location', 'westoutside');
            xlabel('number of iterations');
            hold off;
            %print('../mpaper/figures/parameter_convergence_new', '-depsc2');
        end

        % A wrapper method used for plotting the ELBO and its derivatives in
        % methods below this one
        function obj = run_compute_gradient(obj, parameter_vector)
            obj.gp = vigpirlunpackparam(obj.gp, parameter_vector);
            matrices = vigpirlkernel(obj.gp);
            u_samples = mvnrnd(obj.gp.mu', obj.gp.B * obj.gp.B',...
                obj.algorithm_params.samples_count);
            [elbo, grad] = compute_gradient(obj.mdp_data,...
                obj.demonstrations, obj.counts, obj.gp, u_samples, matrices);

            % Disable derivatives that don't work (if any)
            grad = deactivate_some_derivatives(grad, obj.algorithm_params,...
                obj.d, obj.m);
    
            obj.parameter_history = horzcat(obj.parameter_history, parameter_vector);
            obj.elbo_history = horzcat(obj.elbo_history, elbo);
            obj.grad_history = horzcat(obj.grad_history, grad);
        end

        % Plots that run their own experiments/calculations

        % ELBO as a function of mu over 10 plots
        function elbo_contour_plot(obj)
            mu1_log = [];
            mu2_log = [];
            mu3_log = [];
            mu_range = -40:50:50;
            num_points = length(mu_range) * length(mu_range);
            num_contours = 30;
            parameter_vector = vigpirlpackparam(obj.gp);
            for mu1 = mu_range
                fprintf('.');
                parameter_vector(obj.d + obj.m + 2) = mu1;
                for mu2 = mu_range
                    parameter_vector(obj.d + obj.m + 3) = mu2;
                    for mu3 = mu_range
                        parameter_vector(obj.d + obj.m + 4) = mu3;
                        obj = run_compute_gradient(obj, parameter_vector);
                        mu1_log = [mu1_log mu1];
                        mu2_log = [mu2_log mu2];
                        mu3_log = [mu3_log mu3];
                    end
                end
            end
            fprintf('\n');
            for mu1 = 1:length(mu_range)
                indices = num_points * (mu1 - 1) + 1:num_points * mu1;
                subplot(5, 2, mu1);
                contourf(mu_range, mu_range,...
                    reshape(obj.elbo_history(indices),...
                    [length(mu_range), length(mu_range)]), num_contours);
                xlabel('$\mu_2$', 'Interpreter', 'latex');
                ylabel('$\mu_3$', 'Interpreter', 'latex');
                t = ['$\mu_1 = ', num2str(mu_range(mu1)), '$'];
                title(t, 'Interpreter', 'latex');
            end
        end

        % Policy as a function of rewards over 10 plots
        function policy_contour_plot(obj)
            r1_log = [];
            r2_log = [];
            r3_log = [];
            policy_log = [];
            r1_range = -1:2;
            r_range = -5:5:5;
            num_points = length(r_range) * length(r_range);
            num_contours = 30;
            for r1 = r1_range
                fprintf('.');
                for r2 = r_range
                    for r3 = r_range
                        solution = linearmdpsolve(obj.mdp_data, [r1; r2; r3]);
                        policy = solution.p(1, 1); % state 1, action 1
                        policy_log = [policy_log policy];
                        r1_log = [r1_log r1];
                        r2_log = [r2_log r2];
                        r3_log = [r3_log r3];
                    end
                end
            end
            fprintf('\n');
            for r1 = 1:length(r1_range)
                indices = num_points * (r1 - 1) + (1:num_points);
                subplot(5, 2, r1);
                contourf(r_range, r_range, reshape(policy_log(indices),...
                    [length(r_range), length(r_range)]), num_contours);
                xlabel('$r(s_2)$', 'Interpreter', 'latex');
                ylabel('$r(s_3)$', 'Interpreter', 'latex');
                t = ['$r(s_1) = ', num2str(r1_range(r1)), '$'];
                title(t, 'Interpreter', 'latex');
            end
        end

        % ELBO as a function of mu1=mu2=mu3 for various values of gamma
        function elbo_vs_gamma_plot(obj)
            parameter_vector = vigpirlpackparam(obj.gp);
            mu_range = -10:5;
            gamma_range = 0:0.1:0.9;
            colours = colormap('jet');
            indices = round(linspace(1, length(colours), length(gamma_range)));
            figure('Units', 'centimeters', 'Position', [0 0 15 10],...
                   'PaperPositionMode', 'auto');
            set(groot, 'defaultAxesColorOrder', colours(indices, :));
            hold on;
            for gamma = gamma_range
                obj.mdp_data.discount = gamma;
                mu_log = [];
                for mu = mu_range
                    fprintf('.');
                    parameter_vector((obj.d + obj.m + 2):(obj.d + obj.m + 4)) = mu;
                    obj = obj.run_compute_gradient(parameter_vector);
                    mu_log = [mu_log mu];
                end
                fprintf('\n');
                plot(mu_log, obj.elbo_history((length(obj.elbo_history) -...
                    (length(mu_range) - 1)):end));
            end
            legend('$\gamma = ' + string(gamma_range) + '$', 'Interpreter', 'latex');
            xlabel('$\mu_1 = \mu_2 = \mu_3$', 'Interpreter', 'latex');
            ylabel('$\mathcal{L}$', 'Interpreter', 'latex');
            hold off;
            %print('../mpaper/elbo_over_gamma', '-depsc2');
        end

        % ELBO and its derivative over some element of B
        function elbo_and_derivative_plot(obj)
            figure('Units', 'centimeters', 'Position', [0 0 15 20],...
                'PaperPositionMode', 'auto');

            % B11 (a diagonal element)
            subplot(2, 1, 1);
            B_range = 1:10;
            B_log = [];
            index = obj.d + 2;
            parameter_vector = vigpirlpackparam(obj.gp);
            for B = B_range
                fprintf('.');
                parameter_vector(index) = log(B);
                obj = obj.run_compute_gradient(parameter_vector);
                B_log = [B_log B];
            end
            fprintf('\n');
            parameter_vector(index) = 0;
            first_index = size(obj.grad_history, 2) - length(B_log) + 1;
            xlabel('$B_{1,1}$', 'Interpreter', 'latex');
            yyaxis left;
            plot(B_log, obj.elbo_history(first_index:end));
            ylabel('$E[v]$', 'Interpreter', 'latex');
            yyaxis right;
            plot(B_log, obj.grad_history(index, first_index:end));
            ylabel('$\partial E[v]/\partial B_{1,1}$', 'Interpreter', 'latex');

            % B21 (a non-diagonal element)
            subplot(2, 1, 2);
            B_range = -10:10;
            B_log = [];
            index = obj.d + 2 * obj.m + 2;
            for B = B_range
                fprintf('.');
                parameter_vector(index) = B;
                obj = obj.run_compute_gradient(parameter_vector);
                B_log = [B_log B];
            end
            fprintf('\n');
            parameter_vector(index) = 0;
            first_index = size(obj.grad_history, 2) - length(B_log) + 1;
            xlabel('$B_{2,1}$', 'Interpreter', 'latex');
            yyaxis left;
            plot(B_log, obj.elbo_history(first_index:end));
            ylabel('$E[v]$', 'Interpreter', 'latex');
            yyaxis right;
            plot(B_log, obj.grad_history(index, first_index:end));
            ylabel('$\partial E[v]/\partial B_{2,1}$', 'Interpreter', 'latex');
            %print('../mpaper/elbo_over_B_short', '-depsc2');
        end
    end
end

% A helper function to plot a history matrix. Each row is plotted as a separate sequence.
function plot_history(matrix)
    hold on;
    for row = 1:size(matrix, 1)
        plot(matrix(row,:));
        if (row == 1)
            hold on;
        end
    end
    hold off;
end

% Use algorithm parameters to disable some derivatives
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