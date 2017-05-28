close all;
clear;
clc;

img = imread('banana3.png');

[foreground, coordinates] = imcrop(img);
coordinates = round(coordinates);

foreground_histogram = create_histogram(double(reshape(foreground, [], 3)));

upper_background = img(1:coordinates(2) - 1, :, :);
lower_background = img((coordinates(2) + coordinates(4) + 1):end, :, :);

left_background = img(coordinates(2):(coordinates(2) + coordinates(4)), 1:(coordinates(1) - 1), :);
right_background = img(coordinates(2):(coordinates(2) + coordinates(4)), (coordinates(1) + coordinates(3)+ 1):end, :);

background_color_vectors = [reshape(upper_background, [], 3);
                            reshape(lower_background, [], 3);
                            reshape(left_background, [], 3);
                            reshape(right_background, [], 3)];
                            
background_histogram = create_histogram(double(background_color_vectors));

number_of_gaussians = 5;

pi_components = rand(number_of_gaussians, 1);
pi_components = pi_components / sum(pi_components);








