% Place parameters from specified GP into parameter vector.
function x = vigpirlpackparam(gp)
  x = vertcat(vigpirlhpxform(gp.lambda0, [], 'exp', 3),...
    vigpirlhpxform(gp.lambda, [], 'exp', 3)',...
    vigpirlhpxform(diag(gp.B), [], 'exp', 3), gp.mu, get_lower_triangle(gp.B));
