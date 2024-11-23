function [x, y, k] = center_kernel_separate(x, y, k)
% This code is downloade from http://cs.nyu.edu/~dilip/research/blind-deconvolution/
% Please refer to the following paper for more details:
% Dilip Krishnan, Terence Tay and Rob Fergus, "Blind Deconvolution using a 
% Normalized Sparsity Measure", CVPR 2011 
  
%
% Center the kernel by translation so that boundary issues are mitigated. Additionally,
% if one shifts the kernel the the image must also be shifted in the
% opposite direction.
  
% get centre of mass
mu_y = sum([1:size(k, 1)] .* sum(k, 2)');
mu_x = sum([1:size(k, 2)] .* sum(k, 1));    
  
% get mean offset
offset_x = round( floor(size(k, 2) / 2) + 1 - mu_x );
offset_y = round( floor(size(k, 1) / 2) + 1 - mu_y );

% fprintf('CenterKernel: weightedMean[%f %f] offset[%d %d]\n', mu_x-1, mu_y-1, offset_x, offset_y);

% make kernel to do translation
shift_kernel = zeros(abs(offset_y * 2) + 1, abs(offset_x * 2) + 1);
shift_kernel(abs(offset_y) + 1 + offset_y, abs(offset_x) + 1 + offset_x) = 1;
    
% shift both image and blur kernel
kshift = conv2(k, shift_kernel, 'same');
k = kshift;

xshift = conv2(x, flipud(fliplr(shift_kernel)), 'same');
x = xshift;

yshift = conv2(y, flipud(fliplr(shift_kernel)), 'same');
y = yshift;
