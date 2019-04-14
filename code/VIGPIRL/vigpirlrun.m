% GP-based non-linear IRL algorithm with variational inference
function irl_result = vigpirlrun(algorithm_params, mdp_data, ~,...
                                 feature_data, demonstrations, ~, ~)
  % algorithm_params - parameters of the GP IRL algorithm.
  % mdp_data - definition of the MDP to be solved.
  % demonstrations - cell array containing examples.
  % irl_result - result of IRL algorithm (see bottom of file).

  model = Vigpirl(Mdp(mdp_data, feature_data, demonstrations), algorithm_params);
  [model, time] = model.run();

  matrices = vigpirlkernel(model.gp);
  r = matrices.Kru' * inv(matrices.Kuu) * model.gp.mu;
  solution = linearmdpsolve(mdp_data, r);
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

  irl_result = struct('r', r, 'v', v, 'p', p, 'q', q, 'model_itr', {{model.gp}},...
                      'r_itr', {{r}}, 'model_r_itr', {{r}}, 'p_itr', {{p}},...
                      'model_p_itr', {{p}}, 'time', time, 'score', 0, 'matrices', matrices, 'gp', model.gp);
end

