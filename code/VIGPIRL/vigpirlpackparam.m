% Place parameters from specified GP into parameter vector.
function x = vigpirlpackparam(gp)
  x = gpirlhpxform(gp.rbf_var, [], 'exp', 3);
  x = vertcat(x, gpirlhpxform(gp.inv_widths, [], 'exp', 3)');