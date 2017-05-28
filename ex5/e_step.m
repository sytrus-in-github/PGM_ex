function [ gamma ] = e_step( pis, mus, sigmas, image_vectors )
%E_STEP e-step of EM algorithm for mixture of K guassians on 3D histogram
%of size X,Y,Z. N = X*Y*Z, D=3
%   pis (K) mixture coefficients
%   mus (K,D) center of gaussians
%   sigmas (K,D,D) covariance matrices
%   gamma (K,N)
    k = 5;
    densities = zeros(k, size(image_vectors, 1));
    
    for i=1:k
        density = mvnpdf(image_vectors, mus(:, i)', sigmas(:,:,i));
        densities(i,:) = pis(i) * density;
    end
    gamma = densities ./ sum(densities, 1);
end

