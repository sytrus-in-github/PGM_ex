function [ pis, mus, sigmas ] = m_step( X, gamma )
%M_STEP e-step of EM algorithm for mixture of K guassians on 3D histogram
%of size X,Y,Z. N = X*Y*Z, D=3
%   X (N,D) pixel RGB intensities
%   pis (K) mixture coefficients
%   mus (K,D) center of gaussians
%   sigmas (K,D,D) covariance matrices
%   gamma (K,N)
    n = size(X,1);
    k = size(pis,1);
    Nk = sum(gamma, 2);
    pis = Nk / n;
    mus = gamma * X ./ Nk;
    sigmas = ;
end

