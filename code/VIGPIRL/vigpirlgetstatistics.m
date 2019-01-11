% Pre-compute statistics from examples for GPIRL.
function [mu_sa, init_s] = vigpirlgetstatistics(example_samples, mdp_data)

% Constants.
[states,actions,transitions] = size(mdp_data.sa_s);
[N,T] = size(example_samples);

% Compute expectations.
mu_sa = zeros(states,actions);
init_s = zeros(states,1);
for i=1:N,
    for t=1:T,
        s = example_samples{i,t}(1);
        a = example_samples{i,t}(2);
        
        % Add to state action and state expectations.
        mu_sa(s,a) = mu_sa(s,a) + 1;
        init_s(s) = init_s(s) + 1;
    end;
end;
