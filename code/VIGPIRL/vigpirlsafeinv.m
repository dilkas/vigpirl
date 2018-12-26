% Safe inversion of kernel matrix that uses SVD decomposition if Cholesky
% fails.
function invK = vigpirlsafeinv(K)

% First, try Cholesky.
[L,p] = chol(K,'lower');
if p == 0,
    invK = L'\(L\eye(size(K,1)));
else
    % Must do full SVD decomposition.
    warning('Cholesky failed, switching to SVD');
    [U,S,V] = svd(K);
    dS = diag(S);
    Sinv = diag(1./dS);
    invK = V*Sinv*U';
end;
