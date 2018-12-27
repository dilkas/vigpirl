% Compute likelihood of GP IRL parameters, as well as likelihood gradient.
function [gradient] = vigpirlopt(x, gp, mu_sa, init_s, mdp_data)

% Set constants.
[~,actions] = size(mu_sa);
samples = length(gp.s_u);

[gp,u] = gpirlunpackparam(gp,x);

% Compute kernel and kernel matrix derivatives.
[Kstar,logDetAndPPrior,alpha,Kinv,dhp,dhpdr] = gpirlkernel(gp,u);

% Run value iteration to get policy.
[~, ~, policy, ~] = linearvalueiteration(mdp_data, repmat(r, 1, actions));

if nargout >= 2,    
    % Add hyperparameter prior gradients.
    dhp(1:length(gp.inv_widths)) = dhp(1:length(gp.inv_widths)) + ...
        gpirlhppriorgrad(gp.inv_widths,gp.ard_prior,gp.ard_prior_wt,gp.ard_xform,gp);
    idx = length(gp.inv_widths)+1;
    if gp.warp_x,
        dhp(idx:idx-1+length(gp.inv_widths)) = dhp(idx:idx-1+length(gp.inv_widths)) + ...
            gpirlhppriorgrad(gp.warp_l,gp.warp_l_prior,gp.warp_l_prior_wt,gp.warp_l_xform,gp);
        idx = idx+length(gp.inv_widths);
        dhp(idx:idx-1+length(gp.inv_widths)) = dhp(idx:idx-1+length(gp.inv_widths)) + ...
            gpirlhppriorgrad(gp.warp_c,gp.warp_c_prior,gp.warp_c_prior_wt,gp.warp_c_xform,gp);
        idx = idx+length(gp.inv_widths);
        dhp(idx:idx-1+length(gp.inv_widths)) = dhp(idx:idx-1+length(gp.inv_widths)) + ...
            gpirlhppriorgrad(gp.warp_s,gp.warp_s_prior,gp.warp_s_prior_wt,gp.warp_s_xform,gp);
        idx = idx+length(gp.inv_widths);
    end;
    if gp.learn_noise,
        dhp(idx) = dhp(idx) + ...
            gpirlhppriorgrad(gp.noise_var,gp.noise_prior,gp.noise_prior_wt,gp.noise_xform,gp);
        idx = idx+1;
    end;
    if gp.learn_rbf,
        dhp(idx) = dhp(idx) + ...
            gpirlhppriorgrad(gp.rbf_var,gp.rbf_prior,gp.rbf_prior_wt,gp.rbf_xform,gp);
        idx = idx+1;
    end;
    
end;
