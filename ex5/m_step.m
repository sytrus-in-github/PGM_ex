function [ pis, mus, sigmas ] = m_step( X, gamma )
%M_STEP e-step of EM algorithm for mixture of K guassians on 3D histogram
%of size X,Y,Z. N = X*Y*Z, D=3
%   X (N,D) pixel RGB intensities
%   pis (K) mixture coefficients
%   mus (K,D) center of gaussians
%   sigmas (K,D,D) covariance matrices
%   gamma (K,N)
    d = 3;
    [k, n] = size(gamma);
    Nk = sum(gamma, 2); % size K
    pis = Nk / n;
    mus = gamma * X ./ Nk;
    sigmas = zeros(k, d, d);
    for i=1:k
        X_mean = (X - mus(i,:));
        sigmas(i,:,:) = (X_mean .* gamma(i,:))' * X_mean ./ Nk(i);
    end
end

