function [ normalized_histogram_matrix ] = create_histogram( image_vectors_by_color )
  histogram_matrix = zeros(256, 256, 256);

  for i = 1:size(image_vectors_by_color, 1)
   p = image_vectors_by_color(i, :) + 1;
   histogram_matrix(p(1), p(2), p(3)) = histogram_matrix(p(1), p(2), p(3)) + 1;
  end

  normalized_histogram_matrix = histogram_matrix / sum(histogram_matrix(:));
end