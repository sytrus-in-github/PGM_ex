function [ gamma ] = e_step( pis, mus, sigmas )
%E_STEP e-step of EM algorithm for mixture of K guassians on 3D histogram
%of size X,Y,Z. N = X*Y*Z, D=3
%   pis (K) mixture coefficients
%   mus (K,D) center of gaussians
%   sigmas (K,D,D) covariance matrices
%   gamma (K,N)
    x=256; y=x; z=x;
    k = size(pis,1);
    densities = zeros(k,x,y,z);
    for i=1:k
        densities(i,:,:,:) = pis(i) * gaussian_density(mus(i,:), sigmas(i,:,:), [x,y,z]);
    end
    gamma = reshape(densities, k, []);
    gamma = gamma ./ sum(gamma, 1);
end

