function [ pis_new, mus_new, sigmas_new ] = m_step( histogram, gamma, pis_old, mus_old, sigmas_old )
%M_STEP e-step of EM algorithm for mixture of K guassians on 3D histogram
%of size X,Y,Z. N = X*Y*Z, D=3
%   histogram (X,Y,Z)
%   pis (K) mixture coefficients
%   mus (K,D) center of gaussians
%   sigmas (K,D,D) covariance matrices
%   gamma (K,N)

end

