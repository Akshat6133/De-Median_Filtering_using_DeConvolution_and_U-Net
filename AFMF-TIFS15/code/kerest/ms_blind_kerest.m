function [kernel] = ms_blind_kerest(y, opts)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function is the ms_blind_deconv.m function (yet with some 
% modifications) included in the Matlab code package of their CVPR 2011 
% paper "Blind Deconvolution using a Normalized Sparsity Measure", by 
% Dilip Krishnan, Terence Tay and Rob Fergus.
% 
% Note that the kernel initilization is different from the original.
%
% modified by: Wei FAN (wei.fan@gipsa-lab.grenoble-inp.fr)
% 
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% Do multi-scale blind deconvolution given input file name and options
% structure opts. Returns a double deblurred image along with estimated
% kernel. Following the kernel estimation, a non-blind deconvolution is run.
%
% Copyright (2011): Dilip Krishnan, Rob Fergus, New York University.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% set kernel size for coarsest level - must be odd
minsize = max(3, 2*floor(((opts.kernel_size - 1)/16)) + 1);

% derivative filters
dx = [-1 1; 0 0];
dy = [-1 0; 1 0];

% l2 norm of gradient images
l2norm = 6;

resize_step = sqrt(2); % size ratio of different levels
tmp = minsize;
ksize = [];
while(tmp < opts.kernel_size)
    ksize = cat(1,ksize,tmp);
    tmp = ceil(tmp * resize_step);
    if (mod(tmp, 2) == 0)
        tmp = tmp + 1;
    end
end
ksize = cat(1,ksize,opts.kernel_size);
num_scales = length(ksize); % number of scales

% blind deconvolution - multiscale processing
ks = cell(num_scales,1);
ls = cell(num_scales,1);
for s = 1:num_scales
    if s == 1
        % at coarsest level, initialize kernel
%         ks{s} = init_kernel(ksize(1));
        ks{s} = opts.init_kernel;
    else
        % upsample kernel from previous level to next finer level
        % resize kernel from previous level
        tmp = ks{s-1};
        tmp(tmp<0) = 0; tmp = tmp/sum(tmp(:)); % normalization
        ks{s} = imresize(tmp, [ksize(s) ksize(s)], 'bilinear');
        % bilinear interpolantion not guaranteed to sum to 1 - so renormalize
        ks{s}(ks{s} < 0) = 0; ks{s} = ks{s}./sum(ks{s}(:));
    end
    
    % image size at this level
    r = floor(size(y, 1) * ksize(s) / opts.kernel_size);
    c = floor(size(y, 2) * ksize(s) / opts.kernel_size);
    
    if (s == num_scales)
        r = size(y, 1);
        c = size(y, 2);
    end
    
    % resize y according to the ratio of filter sizes
    ys = imresize(y, [r c], 'bilinear');
    yx = conv2(ys, dx, 'valid');
    yy = conv2(ys, dy, 'valid');
    
    r = min(size(yx, 1), size(yy, 1));
    c = min(size(yx, 2), size(yy, 2));
    
    % the concatenation of the two gradient images
    g = [yx yy];
    
    % normalize to have l2 norm of a certain size
    tmp1 = g(:, 1:c);
    tmp1 = tmp1*l2norm/norm(tmp1(:));
    g(:, 1:c) = tmp1;
    tmp1 = g(:, c+1:end);
    tmp1 = tmp1*l2norm/norm(tmp1(:));
    g(:, c+1:end) = tmp1;
    
    if (s == 1)
        ls{s} = g;
    else
        if (error_flag ~= 0)
            ls{s} = g;
        else
            % upscale the estimated derivative image from previous level
            c1 = (size(ls{s - 1}, 2)) / 2;
            tmp1 = ls{s - 1}(:, 1:c1);
            tmp1_up = imresize(tmp1, [r c], 'bilinear');
            tmp2 = ls{s - 1}(:, c1 + 1 : end);
            tmp2_up = imresize(tmp2, [r c], 'bilinear');
            ls{s} = [tmp1_up tmp2_up];
        end
    end
    
    tmp1 = ls{s}(:, 1:c);
    tmp1 = tmp1*l2norm/norm(tmp1(:));
    ls{s}(:, 1:c) = tmp1;
    tmp1 = ls{s}(:, c+1:end);
    tmp1 = tmp1*l2norm/norm(tmp1(:));
    ls{s}(:, c+1:end) = tmp1;
    
    % call kernel estimation for this scale
    [ls{s}, ks{s}, error_flag] = ss_blind_deconv(g, ls{s}, ks{s}, opts.lambda, opts.delta, opts.x_in_iterN, opts.x_out_iterN, opts.xk_iterN, opts.psi);
    
    if (error_flag < 0)
        ks{s}(:) = 0;
        ks{s}(ceil(size(ks{s}, 1)/2), ceil(size(ks{s}, 2)/2)) = 1;
        fprintf('Bad error - just set output to delta kernel and return\n');
    end
    
    % center the kernel
    c1 = (size(ls{s}, 2)) / 2;
    tmp1 = ls{s}(:, 1:c1);
    tmp2 = ls{s}(:, c1 + 1 : end);
    if sum(isnan(ks{s}))
        tp = zeros(size(ks{s}));
        tp(ceil(end/2)) = 1;
        ks{s} = tp;
    end
    [tmp1_shifted, tmp2_shifted, ks{s}] = center_kernel_separate(tmp1, tmp2, ks{s});
    ls{s} = [tmp1_shifted tmp2_shifted];
    
    % set elements below threshold to 0, and do normalization
    if (s == num_scales)
        kernel = ks{s};
        kernel(kernel(:) < 0) = 0;
        kernel = kernel/sum(kernel(:));
    end
    
end

end