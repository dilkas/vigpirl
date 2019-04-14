classdef Vigpirl
    properties
        algorithm_params
        counts              % Number of times each state appears in the demonstrations
        d                   % Number of features per state
        demonstrations
        gp
        m                   % Number of inducing points
        mdp_data

        elbo_list = [];
        hyperparameter_history = [];
        grad_history = [];
        policy_history = [];
    end
    methods
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

        function [obj, time] = run(obj)
            % Set up variables for dynamic step size algorithms
            num_hyperparameters = obj.m + obj.d + 1 + obj.m*(obj.m+1)/2;
            if strcmp(obj.algorithm_params.step_size_algorithm, 'AdaGrad')
                G = zeros(num_hyperparameters, 1);
            elseif strcmp(obj.algorithm_params.step_size_algorithm, 'AdaDelta')
                E_g = zeros(num_hyperparameters, 1);
                E_x = zeros(num_hyperparameters, 1);
                epsilon = 1e-6;
                rho = 0.6;
            end

            tic;
            for i = 1:obj.algorithm_params.num_iterations
                fprintf('.');
                % Compute the gradient
                matrices = vigpirlkernel(obj.gp);
                z = mvnrnd(obj.gp.mu', obj.gp.B * obj.gp.B', obj.algorithm_params.samples_count);
                [elbo, grad] = full_gradient(obj.mdp_data, obj.demonstrations, obj.counts, obj.gp, z, matrices);
                if (isempty(grad))
                    break;
                end

                old_hyperparameters = vigpirlpackparam(obj.gp);
                grad = deactivate_some_derivatives(grad, obj.algorithm_params, obj.d, obj.m);
                obj.hyperparameter_history = horzcat(obj.hyperparameter_history, old_hyperparameters);

                switch obj.algorithm_params.step_size_algorithm
                case 'AdaGrad'
                    G = G + grad .^ 2;
                    rho = (obj.algorithm_params.learning_rate / sqrt(G))';
                    hyperparameters = old_hyperparameters + rho .* grad;
                case 'AdaDelta'
                    E_g = rho * E_g + (1 - rho) * grad .^ 2;
                    delta = sqrt(E_x + epsilon) ./ sqrt(E_g + epsilon) .* grad;
                    E_x = rho * E_x + (1 - rho) * delta .^ 2;
                    hyperparameters = old_hyperparameters + delta;
                otherwise
                    learning_rate_vector(1:length(grad), 1) = obj.algorithm_params.learning_rate;
                    learning_rate_vector(obj.d + 2:obj.d + obj.m + 1) = obj.algorithm_params.B_learning_rate;
                    learning_rate_vector(obj.d + 2 * obj.m + 2:end) = obj.algorithm_params.B_learning_rate;
                    learning_rate_vector(2) = obj.algorithm_params.lambda1_learning_rate;
                    hyperparameters = old_hyperparameters + learning_rate_vector .* grad;
                end

                obj.gp = vigpirlunpackparam(obj.gp, hyperparameters);

                %fprintf('ELBO: %f\n', elbo);
                obj.elbo_list = horzcat(obj.elbo_list, elbo);
                obj.grad_history = horzcat(obj.grad_history, grad);

                r = matrices.Kru' * inv(matrices.Kuu) * obj.gp.mu;
                solution = linearmdpsolve(obj.mdp_data, r);
                obj.policy_history = horzcat(obj.policy_history, [solution.p(1, 1); solution.p(2, 1); solution.p(3, 2)]);

                if norm(hyperparameters - old_hyperparameters, 1) < obj.algorithm_params.required_precision
                    break;
                end
            end
            fprintf('\n');
            time = toc;
        end

        function convergence_plot(obj)
            %stem(obj.elbo_list);
            plot_history(obj.hyperparameter_history);
            figure();
            plot_history(obj.grad_history);
        end

        function policy_convergence_plot(obj)
            figure('Units', 'centimeters', 'Position', [0 0 12 16], 'PaperPositionMode', 'auto');
            subplot(2, 1, 1);
            stem(obj.elbo_list);
            xlabel('number of iterations');
            ylabel('$\mathcal{L}$', 'Interpreter', 'latex');
            subplot(2, 1, 2);
            plot_history(obj.policy_history);
            xlabel('number of iterations');
            legend('$\pi(a_1 \mid s_1)$', '$\pi(a_1 \mid s_2)$', '$\pi(a_2 \mid s_3)$', 'Interpreter', 'latex', 'Location', 'east');
            %print('../mpaper/figures/convergence_new', '-depsc2');
        end

        function parameter_convergence_plot(obj)
            figure('Units', 'centimeters', 'Position', [0 0 15 5], 'PaperPositionMode', 'auto');
            plot(obj.hyperparameter_history(1,:), 'k-');
            hold on;
            plot(obj.hyperparameter_history(2,:), 'k--');
            plot(obj.hyperparameter_history(6,:), 'b');
            plot(obj.hyperparameter_history(7,:), 'b--');
            plot(obj.hyperparameter_history(8,:), 'b-.');
            plot(exp(obj.hyperparameter_history(3,:)), 'r');
            plot(exp(obj.hyperparameter_history(4,:)), 'r--');
            plot(exp(obj.hyperparameter_history(5,:)), 'r-.');
            plot(obj.hyperparameter_history(9,:), 'g');
            plot(obj.hyperparameter_history(10,:), 'g--');
            plot(obj.hyperparameter_history(11,:), 'g-.');
            legend('$\log \lambda_0$', '$\log \lambda_1$', '$\mu_1$',...
                '$\mu_2$', '$\mu_3$', '$B_{1,1}$', '$B_{2,2}$', '$B_{3,3}$',...
                '$B_{2,1}$', '$B_{3,1}$', '$B_{3,2}$',...
                'Interpreter', 'latex', 'Location', 'westoutside');
            xlabel('number of iterations');
            hold off;
            %print('../mpaper/figures/parameter_convergence_new', '-depsc2');
        end

        function [obj, elbo, grad] = wrapper(obj, parameter_vector)
            % wrapper: vector of parameters -> scalar value * gradient vector
            obj.gp = vigpirlunpackparam(obj.gp, parameter_vector);
            matrices = vigpirlkernel(obj.gp);
            zz = mvnrnd(obj.gp.mu', obj.gp.B * obj.gp.B', obj.algorithm_params.samples_count);
            [elbo, grad] = full_gradient(obj.mdp_data, obj.demonstrations, obj.counts, obj.gp, zz, matrices);

            % Disable derivatives that don't work
            grad = deactivate_some_derivatives(grad, obj.algorithm_params, obj.d, obj.m);
    
            obj.elbo_list = horzcat(obj.elbo_list, elbo);
            obj.hyperparameter_history = horzcat(obj.hyperparameter_history, parameter_vector);
            obj.grad_history = horzcat(obj.grad_history, grad);

            % Turn maximisation into minimisation
            grad = -grad;
            elbo = -elbo;
        end

        % Plots that run their own experiments/calculations

        function elbo_contour_plot(obj)
            % ELBO as a function of mu over 10 plots
            mu1_log = [];
            mu2_log = [];
            mu3_log = [];
            multiplier = 10;
            parameter_vector = vigpirlpackparam(obj.gp);
            for mu1 = 1:10
                real_mu1 = multiplier * (mu1 - 5);
                parameter_vector(obj.d + obj.m + 2) = real_mu1;
                for mu2 = 1:10
                    real_mu2 = multiplier * (mu2 - 5);
                    parameter_vector(obj.d + obj.m + 3) = real_mu2;
                    for mu3 = 1:10
                        real_mu3 = multiplier * (mu3 - 5)
                        parameter_vector(obj.d + obj.m + 4) = real_mu3;
                        obj = wrapper(obj, parameter_vector);
                        mu1_log = [mu1_log real_mu1];
                        mu2_log = [mu2_log real_mu2];
                        mu3_log = [mu3_log real_mu3];
                    end
                end
            end
            for mu1 = 1:10
                indices = 100*(mu1-1)+1:100*mu1;
                other_mu = multiplier * ((1:10) - 5);
                subplot(5, 2, mu1);
                contourf(other_mu, other_mu, reshape(obj.elbo_list(indices), [10, 10]), 30);
                xlabel('$\mu_2$', 'Interpreter', 'latex');
                ylabel('$\mu_3$', 'Interpreter', 'latex');
                t = ['$\mu_1 = ', num2str(multiplier * (mu1 - 5)), '$'];
                title(t, 'Interpreter', 'latex');
            end
        end

        function policy_contour_plot(obj)
            % Policy as a function of rewards over 10 plots
            mu1_log = [];
            mu2_log = [];
            mu3_log = [];
            policy_log = [];
            mu1_range = -4:5;
            mu_range = -5:0.5:5;
            num_points = length(mu_range) * length(mu_range);
            for mu1 = mu1_range
                for mu2 = mu_range
                    for mu3 = mu_range
                        solution = linearmdpsolve(obj.mdp_data, [mu1; mu2; mu3]);
                        policy = solution.p(1, 1); % state 1, action 1
                        policy_log = [policy_log policy];
                        mu1_log = [mu1_log mu1];
                        mu2_log = [mu2_log mu2];
                        mu3_log = [mu3_log mu3];
                    end
                end
            end
            for mu1 = 1:length(mu1_range)
                indices = num_points * (mu1-1) + (1:num_points);
                subplot(5, 2, mu1);
                contourf(mu_range, mu_range, reshape(policy_log(indices), [length(mu_range), length(mu_range)]), 30);
                xlabel('$r(s_2)$', 'Interpreter', 'latex');
                ylabel('$r(s_3)$', 'Interpreter', 'latex');
                t = ['$r(s_1) = ', num2str(mu1 - 5), '$'];
                title(t, 'Interpreter', 'latex');
            end
        end

        function elbo_vs_gamma_plot(obj)
            % ELBO as a function of mu for various values of gamma
            parameter_vector = vigpirlpackparam(obj.gp);
            mu_from = -10;
            mu_to = 5;
            gamma_from = 0;
            gamma_to = 9;
            colours = colormap('jet');
            indices = round(linspace(1, length(colours), gamma_to - gamma_from + 1));
            figure('Units', 'centimeters', 'Position', [0 0 15 10],...
                   'PaperPositionMode', 'auto');
            set(groot, 'defaultAxesColorOrder', colours(indices, :));
            hold on;
            for gamma = gamma_from:gamma_to
                obj.mdp_data.discount = gamma / 10.0;
                mu_log = [];
                for mu = mu_from:mu_to
                    fprintf('.');
                    parameter_vector(obj.d + obj.m + 2:obj.d + obj.m + 4) = mu;
                    obj = obj.wrapper(parameter_vector);
                    mu_log = [mu_log mu];
                end
                fprintf('\n');
                plot(mu_log, obj.elbo_list((length(obj.elbo_list)-(mu_to - mu_from)):end));
            end
            legend('$\gamma = ' + string((gamma_from:gamma_to) / 10.0) + '$', 'Interpreter', 'latex');
            xlabel('$\mu_1 = \mu_2 = \mu_3$', 'Interpreter', 'latex');
            ylabel('$\mathcal{L}$', 'Interpreter', 'latex');
            hold off;
            %print('../mpaper/elbo_over_gamma', '-depsc2');
        end

        function elbo_and_derivative_plot(obj)
            % ELBO and derivative over some element of B
            figure('Units', 'centimeters', 'Position', [0 0 15 20], 'PaperPositionMode', 'auto');
            subplot(2, 1, 1);
            accuracy = 1;
            B_from = 1;
            B_to = 10;
            B_log = [];
            index = obj.d + 2;
            parameter_vector = vigpirlpackparam(obj.gp);
            for B = B_from:accuracy:B_to
                fprintf('.');
                parameter_vector(index) = log(B);
                obj = obj.wrapper(parameter_vector);
                B_log = [B_log B];
            end
            fprintf('\n');
            parameter_vector(index) = 0;
            first_index = size(obj.grad_history, 2) - length(B_log) + 1;
            xlabel('$B_{1,1}$', 'Interpreter', 'latex');
            yyaxis left;
            plot(B_log, obj.elbo_list(first_index:end));
            ylabel('$E[v]$', 'Interpreter', 'latex');
            yyaxis right;
            plot(B_log, obj.grad_history(index, first_index:end));
            ylabel('$\partial E[v]/\partial B_{1,1}$', 'Interpreter', 'latex');

            subplot(2, 1, 2);
            B_from = -10;
            B_to = 10;
            accuracy = 1;
            B_log = [];
            index = obj.d + 2 * obj.m + 2;
            for B = B_from:accuracy:B_to
                fprintf('.');
                parameter_vector(index) = B;
                obj = obj.wrapper(parameter_vector);
                B_log = [B_log B];
            end
            fprintf('\n');
            parameter_vector(index) = 0;
            first_index = size(obj.grad_history, 2) - length(B_log) + 1;
            xlabel('$B_{2,1}$', 'Interpreter', 'latex');
            yyaxis left;
            plot(B_log, obj.elbo_list(first_index:end));
            ylabel('$E[v]$', 'Interpreter', 'latex');
            yyaxis right;
            plot(B_log, obj.grad_history(index, first_index:end));
            ylabel('$\partial E[v]/\partial B_{2,1}$', 'Interpreter', 'latex');
            %print('../mpaper/elbo_over_B_short', '-depsc2');
        end
    end
end

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