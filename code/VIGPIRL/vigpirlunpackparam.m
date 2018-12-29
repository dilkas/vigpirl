% Place parameters from parameter vector into GP.
function [gp] = vigpirlunpackparam(gp, x)

% Count the last index read.
endi = length(x);

gp.rbf_var = gpirlhpxform(x(endi), [], 'exp', 1);
endi = endi - 1;

% Read ARD kernel parameters.
gp.inv_widths = gpirlhpxform(x(endi-length(gp.inv_widths)+1:endi), [], 'exp', 1);
endi = endi - length(gp.inv_widths);

gp.mu = x(endi-length(gp.mu)+1:endi);
endi = endi - length(gp.mu);