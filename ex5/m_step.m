function [ pis, mus, sigmas ] = m_step( X, gamma )
%M_STEP e-step of EM algorithm for mixture of K guassians on 3D histogram
%of size X,Y,Z. N = X*Y*Z, D=3
%   X (N,D) pixel RGB intensities
%   pis (K) mixture coefficients
%   mus (D,K) center of gaussians
%   sigmas (D,D,K) covariance matrices
%   gamma (K,N)
    d = 3;
    [k, n] = size(gamma);
    Nk = sum(gamma, 2); % size K
    pis = (Nk / n)';
    mus = (gamma * X ./ Nk)';
    sigmas = zeros(d, d, k);
    for i=1:k
        X_mean = (X - mus(:, i)');
        g = gamma(i, :);
        neg_count = sum(g(g<0));
        if neg_count > 0
          frpintf('Error!!, %d negative', neg_count);
        end
        
        sigmas(:,:,i) = (X_mean .* gamma(i,:)')' * X_mean ./ Nk(i);
        sigmas(:,:,i) = (sigmas(:,:,i) + sigmas(:,:,i)') / 2 + eye(d) * 1e-8;
    end
end

