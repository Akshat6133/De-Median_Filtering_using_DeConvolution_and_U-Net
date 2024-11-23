%MFDCONV.m
function [ dmfI ] = MFDCONV( IMG, opts )
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%MFDCONV processes a median filtered image using the proposed image
% variational deconvolution method, for both MF image quality enhancement
% and anti-forensic purposes
%
%   INPUT   
%           IMG: image name / image pixel value matrix
%          opts: parameter settings
% 
%   OUTPUT
%          dmfI: the processed MF anti-forensic image
% 
% -------------------------------------------------------------------------
% Copyright (c) 2015 GIPSA-Lab/Grenoble INP, and Beihang University
% All Rights Reserved.
% -------------------------------------------------------------------------
% Permission to use, copy, or modify this software and its documentation
% for educational and research purposes only and without fee is hereby
% granted, provided that this copyright notice and the original authors'
% names appear on all copies and supporting documentation. This program
% shall not be used, rewritten, or adapted as the basis of a commercial
% software or hardware product without first obtaining permission of the
% authors. The authors make no representations about the suitability of
% this software for any purpose. It is provided "as is" without express
% or implied warranty.
% -------------------------------------------------------------------------
% If you find any bugs, please kindly report to us.
% -------------------------------------------------------------------------
%
% contact:       wei.fan@gipsa-lab.grenoble-inp.fr
% last modified: May. 9th, 2014
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if ischar(IMG)
    M = double(imread(IMG)); % read image file
else
    M = double(uint8(IMG)); % image pixel value matrix
end

if opts.winS ~= 3
    error('This function currently only supports the window size of 3!');
end

if any(strcmp('kernel',fieldnames(opts))) && ~isempty(opts.kernel)
    kernel = opts.kernel;
else
    % multi-scale blind convolution kernel estimation
    opts.kernel_size = opts.winS;
    opts.delta = 0.01;
    opts.x_in_iterN = 2;
    opts.x_out_iterN = 2; % only two iterations
    opts.xk_iterN = 2;
    opts.lambda = 100;
    opts.psi = 0;
    kernel = ms_blind_kerest(M, opts);
end

%%% estimate parameters of the generalized Gaussian distribution
load mfpemf3GGPElrCoefs.mat; mfGGPElrCoefs = mfpemf3GGPElrCoefs;
[opts.nb_alphas.h1,opts.nb_betas.h1] = estGGParams(M, opts.winS, [1 -1], mfGGPElrCoefs.var.h1, mfGGPElrCoefs.kurt.h1);
[opts.nb_alphas.v1,opts.nb_betas.v1] = estGGParams(M, opts.winS, [1 -1]', mfGGPElrCoefs.var.v1, mfGGPElrCoefs.kurt.v1);
[opts.nb_alphas.h2,opts.nb_betas.h2] = estGGParams(M, opts.winS, [1 0 -1], mfGGPElrCoefs.var.h2, mfGGPElrCoefs.kurt.h2);
[opts.nb_alphas.v2,opts.nb_betas.v2] = estGGParams(M, opts.winS, [1 0 -1]', mfGGPElrCoefs.var.v2, mfGGPElrCoefs.kurt.v2);

%%% apply non-blind deconvolution
% padding
bhs = floor(opts.winS/2);
padI = padarray(M, [bhs bhs], 'replicate', 'both');
for i = 1:4
    padI = edgetaper(padI, kernel);
end
% non-blind deconvolution
tmp = nblind_deconv_gau_bregman(padI, kernel, opts.nb_lambda, opts.nb_gamma, opts.nb_omega, opts.nb_alphas, opts.nb_betas, opts.iDither);
dmfI = tmp(bhs + 1 : end - bhs, bhs + 1 : end - bhs);
% replace the border of the resulting image using the original one
halfWinS = ceil(opts.winS/2);
dmfI([1:halfWinS end-halfWinS+1:end],:) = M([1:halfWinS end-halfWinS+1:end],:);
dmfI(:,[1:halfWinS end-halfWinS+1:end]) = M(:,[1:halfWinS end-halfWinS+1:end]);
% rounding and truncation
dmfI = double(uint8(dmfI));

end

function [ alpha, beta ] = estGGParams(M1, winS, filt, coefVar, coefKurt)
% for estimating the parameters for the generalized Gaussian distribution

M2 = double(medfilt2(M1,[winS,winS],'symmetric'));
M3 = double(medfilt2(M2,[winS,winS],'symmetric'));
M4 = double(medfilt2(M3,[winS,winS],'symmetric'));
M5 = double(medfilt2(M4,[winS,winS],'symmetric'));

G1 = conv2(M1, filt, 'valid'); varM1 = var(G1(:)); kurtM1 = kurtosis(G1(:));
G2 = conv2(M2, filt, 'valid'); varM2 = var(G2(:)); kurtM2 = kurtosis(G2(:));
G3 = conv2(M3, filt, 'valid'); varM3 = var(G3(:)); kurtM3 = kurtosis(G3(:));
G4 = conv2(M4, filt, 'valid'); varM4 = var(G4(:)); kurtM4 = kurtosis(G4(:));
G5 = conv2(M5, filt, 'valid'); varM5 = var(G5(:)); kurtM5 = kurtosis(G5(:));

varVec = [1,varM1,varM2,varM3,varM4,varM5];
kurtVec = [1,kurtM1,kurtM2,kurtM3,kurtM4,kurtM5];

varEst = varVec*coefVar;
kurtEst = kurtVec*coefKurt;

betas = 0.001:0.001:2;
kurts = gamma(1./betas).*gamma(5./betas)./(gamma(3./betas).^2);
[~,ind] = min(abs(kurts-kurtEst));
beta = abs(betas(ind));
alpha = abs(sqrt( varEst*gamma(1/beta)/gamma(3/beta) ));

end