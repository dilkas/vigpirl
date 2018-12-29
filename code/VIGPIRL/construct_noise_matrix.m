function [nmat, nmat2, nconst] = construct_noise_matrix(n, noise, inv_widths)
  nconst = exp(-0.5 * noise * sum(inv_widths));
  nconst2 = -0.5 * noise;
  nmat = nconst * ones(n) + (1 - nconst) * eye(n);
  nmat2 = nconst2 * ones(n) - nconst2 * eye(n);
end
