% Place parameters from parameter vector into GP.
function [gp] = vigpirlunpackparam(gp, x)

% Count the last index read.
endi = length(x);
m = length(gp.mu);

gp.rbf_var = vigpirlhpxform(x(endi), [], 'exp', 1);
endi = endi - 1;

% Read ARD kernel parameters.
gp.inv_widths = vigpirlhpxform(x(endi-length(gp.inv_widths)+1:endi), [], 'exp', 1)';
endi = endi - length(gp.inv_widths);

gp.mu = x(endi-m+1:endi);
endi = endi - m;

gp.B(1:m+1:end) = vigpirlhpxform(x(endi-m+1:endi), [], 'exp', 1)';
endi = endi - m;