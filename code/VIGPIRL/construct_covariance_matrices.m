% Construct two covariance matrices: Gamma (the covariance matrix of r|u) and
% the posterior covariance matrix of r, combining the distributions of u and r|u
function [covariance, Gamma] = construct_covariance_matrices(result)
    S = result.matrices.Kru' * inv(result.matrices.Kuu);
    Gamma = result.matrices.Krr - S * result.matrices.Kru;
    covariance = Gamma + S * result.gp.B * result.gp.B' * S';
end