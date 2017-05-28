function [ pis, mus, covariances ] = initialize_parameters( number_of_components, dimension)
  pis = rand(number_of_components, 1);
  pis = pis' / sum(pis);
  
  mu_lower = 1;
  mu_upper = 256;
  mus = (mu_upper - mu_lower) * rand(dimension, number_of_components) + mu_lower;
  
  covariances = reshape(repmat(eye(dimension), 1, number_of_components), ...
                 [dimension, dimension, number_of_components]); 
end

