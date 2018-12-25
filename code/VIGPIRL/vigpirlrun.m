                                % GP-based non-linear IRL algorithm.
function irl_result = vigpirlrun(algorithm_params,mdp_data,mdp_model,...
                                 feature_data,example_samples,~,verbosity)

                    % algorithm_params - parameters of the GP IRL algorithm.
                    % mdp_data - definition of the MDP to be solved.
                    % example_samples - cell array containing examples.
                    % irl_result - result of IRL algorithm (see bottom of file).

                                % Fill in default parameters.
  algorithm_params = vigpirldefaultparams(algorithm_params);

                                % Set random seed.
  rand('seed',algorithm_params.seed);
  randn('seed',algorithm_params.seed);

                      % Get state-action counts and initial state distributions.
  [mu_sa,init_s] = vigpirlgetstatistics(example_samples,mdp_data);

                                % Set up optimization options.
  options = struct('Display','iter','LS_init',2,'LS',2,'Method','lbfgs',...
                   'MaxFunEvals',4000,'MaxIter',2000);
  if verbosity < 2,
    options.display = 'none';
  end;

                                % Create GP.
  if ~isempty(algorithm_params.initial_gp),
    gp = algorithm_params.initial_gp;
  else
                                % Create initial GP.
    gp = gpirlinit(algorithm_params,feature_data);
                                % Choose inducing points.
    gp = vigpirlgetinducingpoints(gp,mu_sa,algorithm_params);
  end;

  m = size(gp.X_u, 1);
  d = size(feature_data.splittable, 2);

                                % Randomly initialize variational parameters
                                % TODO: do the optimal thing instead
  mu = rand(m, 1);
  lambda0 = random('Chisquare', algorithm_params.rbf_init);
  lambda = random('Chisquare', algorithm_params.ard_init, [d, 1]);

                 % L is a lower triangular matrix with positive diagonal entries
  L = normrnd(0, 1, [m, m]);
  L(1:m+1:end) = random('Chisquare', 1, [m, 1]);
  L = tril(L);

  function fun = estimate_derivative(solution, Kuu_inv, Kuu_derivative)
                                % TODO: make this return a function that takes u
    function derivative = wrapped(u)
      function a = inner(state)
        a = solution.v(state, 1) * trace((Kuu_inv * u * u' * Kuu_inv' - Kuu_inv) * Kuu_derivative);
      end;

      function answer = foo(state_action)
        s = 0;
        for next_state = 1:size(mdp_data.sa_p, 3),
          s = s + mdp_data.sa_p(state_action(1), state_action(2), next_state) * inner(next_state);
        end;
        answer = inner(state_action(1)) - mdp_data.discount * s;
      end;

      derivative = sum(sum(cellfun(@foo, example_samples)));
    end;

    fun = @wrapped;
  end;

  tic;
  % TODO: lambda0 should not be negative. Exp?
  for i = 1:100,
                 % TODO: use more than one sample
                 % Draw samples_count samples from the variational approximation
    disp(lambda0);
    u = mvnrnd(mu, L * L')';
    rho = 1; % TODO: implement AdaGrad from BBVI
    gp.Y = u; % TODO: probably not needed
    [Kru, ~, Kuu_inv, Kuu, KruKuu, Krr, Kuu_derivative, Kru_derivative] = vigpirlkernel(gp, u);
    r = mvnrnd(KruKuu * u, Krr - KruKuu * Kru)';
    solution = feval([mdp_model 'solve'], mdp_data, r);
    estimator = estimate_derivative(solution, Kuu_inv, Kuu_derivative);
    lambda0 = lambda0 + rho * (-0.5 * trace(Kuu_inv * Kuu_derivative)...
                               + init_s' * (Kru_derivative' - KruKuu * Kuu_derivative)...
                                 * Kuu_inv * mu - 0.5 * estimator(u));
  end;
  time = toc;
                                % TODO: stopping condition

                           % Uncomment the following line to enable derivatives.
                           %options.DerivativeCheck = 'on';

      % Since we will be running multiple iterations, start with high tolerance.
  options.TolFun = algorithm_params.restart_tolerance;

                 % Now run additional restarts to get kernel parameters correct.
  if gp.warp_x,
                                % Doing random restarts.
    iterations = algorithm_params.warp_x_restarts;
  else
                         % Restart is deterministic, so no need for more than 2.
    iterations = 1;
  end;

                                % Return corresponding reward function.
  v = solution.v;
  q = solution.q;
  p = solution.p;

                                % Construct returned structure.
                                % TODO: score = best elbo
  irl_result = struct('r',r,'v',v,'p',p,'q',q,'model_itr',{{gp}},...
                      'r_itr',{{r}},'model_r_itr',{{r}},'p_itr',{{p}},'model_p_itr',{{p}},...
                      'time',time,'score', 0);
end
