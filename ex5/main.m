close all;
clear;
clc;

img = imread('banana3.png');

K = 5;

[foreground, coordinates] = imcrop(img);
coordinates = round(coordinates);

img_vectors = double(reshape(img, [], 3));

foreground_vectors = double(reshape(foreground, [], 3));

upper_background = img(1:coordinates(2) - 1, :, :);
lower_background = img((coordinates(2) + coordinates(4) + 1):end, :, :);

left_background = img(coordinates(2):(coordinates(2) + coordinates(4)), 1:(coordinates(1) - 1), :);
right_background = img(coordinates(2):(coordinates(2) + coordinates(4)), (coordinates(1) + coordinates(3)+ 1):end, :);

background_vectors = double([reshape(upper_background, [], 3);
                            reshape(lower_background, [], 3);
                            reshape(left_background, [], 3);
                            reshape(right_background, [], 3)]);

[pis_f, mus_f, sigmas_f] = execute_em(foreground_vectors);
[pis_b, mus_b, sigmas_b] = execute_em(background_vectors);

pdf_f = zeros(size(img_vectors, 1), 1);
pdf_b = zeros(size(img_vectors, 1), 1);

for i = 1:K
   mixture_coefficient_f = pis_f(i);
   mu_f = mus_f(:, i);
   sigma_f = sigmas_f(:,:,i);
   
   temp_pdf_f = mixture_coefficient_f * mvnpdf(img_vectors, mu_f', sigma_f);
   
   pdf_f = pdf_f + temp_pdf_f;
   
   mixture_coefficient_b = pis_b(i);
   mu_b = mus_b(:, i);
   sigma_b = sigmas_b(:,:,i);
   
   temp_pdf_b = mixture_coefficient_b * mvnpdf(img_vectors, mu_b', sigma_b);
   pdf_b = pdf_b + temp_pdf_b;
end

 labels = pdf_f > pdf_b;
  
 segmentation = reshape(labels, size(img, 1), size(img, 2));
 
 figure
 imshow(uint8(segmentation) * 255)
 
%                             
% background_histogram = create_histogram(double(background_color_vectors));
% 
% number_of_gaussians = 5;
% 
% pi_components = rand(number_of_gaussians, 1);
% pi_components = pi_components / sum(pi_components);