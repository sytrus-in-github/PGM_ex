function [ pis, mus, sigmas ] = initialize_parameters( number_of_components, dimension, img_vectors)

  pis = ones(1, number_of_components) / number_of_components;
  
%   mus = (mu_upper - mu_lower) * rand(dimension, number_of_components) + mu_lower;
  
  mus = initialize_mus(img_vectors, number_of_components, dimension);
  sigmas = 50 * reshape(repmat(eye(dimension), 1, number_of_components), ...
                 [dimension, dimension, number_of_components]); 
                        
end

function [mus] = initialize_mus(img_vectors, number_of_components, dimension)
  mus = zeros(dimension, number_of_components);
  
  for i = 1:dimension
    vector = img_vectors(:, i);
    max_value = max(vector);
    min_value = min(vector);
  
    mus(i, :) = (max_value - min_value) * rand(1, number_of_components) + min_value;
  end
end