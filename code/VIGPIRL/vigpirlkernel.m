                      % Optimized kernel computation function for DC mode GPIRL.
function [K_uf, invK, K_uu, K_ufKinv, K_ff, K_uu_deriv_lambda0, K_uf_deriv_lambda0] = vigpirlkernel(gp,Xstar)

                                % Constants.
  dims = length(gp.inv_widths);
  n = size(gp.X_u,1);

                                                            % Undo transforms.
  inv_widths = gpirlhpxform(gp.inv_widths,[],gp.ard_xform,1); % This is \Lambda
  noise_var = gpirlhpxform(gp.noise_var,[],gp.noise_xform,1); % This is 2\sigma^2
  rbf_var = gpirlhpxform(gp.rbf_var,[],gp.rbf_xform,1); % This is \beta
  if gp.warp_x,
    warp_c = gpirlhpxform(gp.warp_c,[],gp.warp_c_xform,1); % This is m
    warp_l = gpirlhpxform(gp.warp_l,[],gp.warp_l_xform,1); % This is \ell
    warp_s = gpirlhpxform(gp.warp_s,[],gp.warp_s_xform,1); % This is s
  end;
  inv_widths = min(inv_widths,1e100); % Prevent overflow.

                                % Compute scales.
  iw_sqrt = sqrt(inv_widths);

                                % Scale positions in feature space.
  if gp.warp_x,
    [X_u_warped,dxu] = gpirlwarpx(gp.X_u,warp_c,warp_l,warp_s);
    [X_f_warped,dxf] = gpirlwarpx(gp.X,warp_c,warp_l,warp_s);
    if nargin >= 2,
      [X_s_warped,dxs] = gpirlwarpx(Xstar,warp_c,warp_l,warp_s);
    end;
  else
    dxf = [];
    X_u_warped = gp.X_u;
    X_f_warped = gp.X;
    if nargin >= 2,
      X_s_warped = Xstar;
    end;
  end;
  X_u_scaled = bsxfun(@times,iw_sqrt,X_u_warped);
  X_f_scaled = bsxfun(@times,iw_sqrt,X_f_warped);

  function [nmat, nconst, dxu_ssum] = construct_noise_matrix(n)
    mask_mat = ones(n)-eye(n);
    nconst = exp(-0.5*noise_var*sum(inv_widths));
    if gp.warp_x,
                                % Noise is spatially varying.
      dxu_scaled = -0.25*noise_var*bsxfun(@times,inv_widths,dxu);
      dxu_ssum = sum(dxu_scaled,2);
      nudist = bsxfun(@plus,dxu_ssum,dxu_ssum');
      nudist(~mask_mat) = 0;
      nmat = exp(nudist);
    else
                                % Noise is uniform.
      dxu_ssum = [];
      nmat = nconst*ones(n) + (1-nconst)*eye(n);
    end;
  end;

  function [K_uu, nconst, dxu_ssum] = compute_covariance_matrix(X, diff_by_lambda0)
    d_uu = bsxfun(@plus,sum(X.^2,2),sum(X.^2,2)') - 2 * X * X';
    d_uu = max(d_uu,0);
    [nmat, nconst, dxu_ssum] = construct_noise_matrix(size(X, 1));
    K_uu = exp(-0.5*d_uu).*nmat;
    if ~diff_by_lambda0,
      K_uu = rbf_var * K_uu;
    end;
  end;

  [K_ff, ~, ~] = compute_covariance_matrix(X_f_scaled, false);
  [K_uu, nconst, dxu_ssum] = compute_covariance_matrix(X_u_scaled, false);
  [K_uu_deriv_lambda0, ~, ~] = compute_covariance_matrix(X_u_scaled, true);

  function K_uf = compute_uf_matrix(X_f_scaled, dxf, diff_by_lambda0)
    d_uf = bsxfun(@plus,sum(X_u_scaled.^2,2),sum(X_f_scaled.^2,2)') - 2*(X_u_scaled*(X_f_scaled'));
    d_uf = max(d_uf,0);
    if gp.warp_x,
                                % Noise is spatially varying.
      dxf_scaled = -0.25*noise_var*bsxfun(@times,inv_widths,dxf);
      dxf_ssum = sum(dxf_scaled,2);
      nfdist = bsxfun(@plus,dxu_ssum,dxf_ssum');
      K_uf = exp(-0.5*d_uf).*exp(nfdist);
      if ~diff_by_lambda0,
        K_uf = rbf_var * K_uf;
      end;
    else
                                % Noise is uniform.
      K_uf = nconst*rbf_var*exp(-0.5*d_uf);
    end;
  end;

  % TODO: could reduce this (and similar pieces of code) to a single function call
  if nargin < 2,
    K_uf = compute_uf_matrix(X_f_scaled, dxf, false);
    K_uf_deriv_lambda0 = compute_uf_matrix(X_f_scaled, dxf, true);
  else
                                % Use Xstar to compute K_uf matrix.
    X_s_scaled = bsxfun(@times,iw_sqrt,X_s_warped);
    K_uf = compute_uf_matrix(X_s_scaled, dxs, false);
    K_uf_deriv_lambda0 = compute_uf_matrix(X_s_scaled, dxs, true);
  end;

                                % Invert the kernel matrix.
  try
    invK = vigpirlsafeinv(K_uu);
  catch err
                                % Display the error.
    rethrow(err);
  end;
  K_ufKinv = K_uf'*invK;
end
