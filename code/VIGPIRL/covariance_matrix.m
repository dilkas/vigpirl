function [covariance, Gamma] = covariance_matrix(result)
    S = result.matrices.Kru' * inv(result.matrices.Kuu);
    Gamma = result.matrices.Krr - S * result.matrices.Kru;
    covariance = Gamma + S * result.gp.B * result.gp.B' * S';
end