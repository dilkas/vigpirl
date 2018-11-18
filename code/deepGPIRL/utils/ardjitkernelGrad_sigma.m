function grad_sigma = ardjitkernelGrad_sigma(X,Y,Lambda,beta,sigma)
% TESTED!
iw_sqrt = sqrt(Lambda);
N = size(X,1);
if ~isempty(Y)
    X_scaled = bsxfun(@times,iw_sqrt,X);
    Y_scaled = bsxfun(@times,iw_sqrt,Y);
    d_uf = bsxfun(@plus,sum(X_scaled.^2,2),sum(Y_scaled.^2,2)') - 2*(X_scaled*(Y_scaled'));
    d_uf = max(d_uf,0);
    nconst = exp(-0.5*sigma*sum(Lambda));
    KXY = nconst*beta*exp(-0.5*d_uf); % Noise is uniform.
    grad_sigma = -0.5*sum(Lambda)*KXY;
else
    X_scaled = bsxfun(@times,iw_sqrt,X);
    d_uu = bsxfun(@plus,sum(X_scaled.^2,2),sum(X_scaled.^2,2)') - 2*(X_scaled*(X_scaled'));
    d_uu = max(d_uu,0);
    nconst = exp(-0.5*sigma*sum(Lambda));
    nmat = nconst*ones(size(X,1)) + (1-nconst)*eye(size(X,1));
    KXX = beta*exp(-0.5*d_uu).*nmat; 
    grad_sigma = -0.5*sum(Lambda)*KXX .* (ones(N)-eye(N));
end
