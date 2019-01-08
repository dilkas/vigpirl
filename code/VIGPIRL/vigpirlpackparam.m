% Place parameters from specified GP into parameter vector.
function x = vigpirlpackparam(gp)
  x = vigpirlhpxform(gp.rbf_var, [], 'exp', 3);
  x = vertcat(x, vigpirlhpxform(gp.inv_widths, [], 'exp', 3)', gp.mu,...
    vigpirlhpxform(diag(gp.B), [], 'exp', 3), get_lower_triangle(gp.B));
