function [ density_tensor ] = gaussian_density( mu, sigma, shape )
  %GAUSSIAN_DENSITY return normaized gaussian density in a 3D grid
  %   mu (D) center of gaussian
  %   sigma (D,D) covariance matrix
  %   shape (3) shape of 3D grid

  sigma_inv = inv(sigma);
  normalizer = 1 / sqrt(det(2*pi*sigma));
  
  density_tensor = zeros(shape);
  
  for i = 1:shape(1)
    for j = 1:shape(2)
      for k = 1:shape(3)
         x = [i; j; k];
         density = exp(-0.5 * (x - mu)' * sigma_inv * (x - mu)) * normalizer ;
         density_tensor(i, j, k) = density;
      end
    end
  end
  
  density_tensor = density_tensor / sum(density_tensor(:));
end