function [ pis, mus, sigmas] = execute_em( image_vectors )
   K = 5;
   dim = 3;
   [pis, mus, sigmas] = initialize_parameters(K, dim, image_vectors);
   
   for i = 1:10000
     gammas = e_step(pis, mus, sigmas, image_vectors);
     [pis_new, mus_new, sigmas_new] = m_step(image_vectors, gammas);
     
     do_print = mod(i, 10) == 0;
     if do_print
       fprintf('Iteration number: %d\n', i);
     end
     
     if (converged({pis, mus, sigmas}, {pis_new, mus_new, sigmas_new}, do_print))
       break;
     end
     
     pis = pis_new;
     mus = mus_new;
     sigmas = sigmas_new;
   end 
%EXECUTE_EM Summary of this function goes here
%   Detailed explanation goes here
end

function [is_converged] = converged(old_params, new_params, do_print)
   pis_diff = abs(old_params{1} - new_params{1});
   mus_diff = abs(old_params{2} - new_params{2});
   sigmas_diff = abs(old_params{3} - new_params{3});
   
   max_diff_value = max([max(pis_diff(:)) max(mus_diff(:)) max(sigmas_diff(:))]);
   if do_print
     fprintf('Diff between iter: %f\n', max_diff_value);
   end
   is_converged =  max_diff_value < 0.1;
end
