close all;
clear;
clc;

img = imread('banana3.png');

[foreground, coordinates] = imcrop(img);
coordinates = round(coordinates);

foreground_vectors_by_color = double(reshape(foreground, [], 3));

histogram_matrix = zeros(256, 256, 256);

for i = 1:size(foreground_vectors_by_color, 1)
   p = foreground_vectors_by_color(i, :) + 1;
   histogram_matrix(p(1), p(2), p(3)) = histogram_matrix(p(1), p(2), p(3)) + 1;
end

normalized_histogram_matrix = histogram_matrix / sum(histogram_matrix(:));

number_of_gaussians = 5;

pi_components = rand(number_of_gaussians, 1);
pi_components = pi_components / sum(pi_components);








