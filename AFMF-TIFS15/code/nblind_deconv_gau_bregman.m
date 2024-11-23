%nblind_deconv_gau_bregman.m

function [xhat] = nblind_deconv_gau_bregman(y, k, lambda, gamma, omega, alphas, betas, iDither)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%nblind_deconv_gau_bregman solve the non-blind deconvolution problem using 
% the Bregman iteration
%
%   INPUT   
%             y: the degraded image
%             k: kernel matrix
%        lambda: paramter in front of the image fidelity term
%         gamma: paramter in front of the splitted quandratic term
%        alphas: scale parameter of the GG
%         betas: shape parameter of the GG
%       iDither: whether to slightly dither the image before further processing
% 
%   OUTPUT
%          xhat: output image
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
% 
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% This function is modified from (inspired by) the fast_deconv_bregman.m 
% function included in the Matlab code package of their CVPR 2011 paper 
% "Blind Deconvolution using a Normalized Sparsity Measure", by 
% Dilip Krishnan, Terence Tay and Rob Fergus.
% 
% Also note that our problem is quite different from theirs, and the
% modification to their original code is large. Please refer to our paper 
% for the details of our optimization problem.
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% make sure k is odd-sized
if ((mod(size(k,1), 2) ~= 1) || (mod(size(k,2), 2) ~= 1) || size(k,1) ~= size(k,2))
    error('Error - blur kernel k must be odd-sized and square.\n');
else
    winS = size(k,1);
end

xhat = y; % initialize
if nargin >= 7 && iDither
    % dithering
    xhat = ditherMFI(xhat, winS);
    xhat = ditherMFI(xhat', winS)';
end

% calculate the constants
sizef = size(y);
otfk  = psf2otf(k, sizef);
i = zeros(winS); i(ceil(end/2)) = 1;
otfi  = psf2otf(i, sizef);
Kty = conj(otfk) .* fft2(y);
KtK = abs(otfk) .^ 2;
ItI = abs(otfi) .^ 2;
FtF = zeros(size(KtK));

% initialization, calculattion of the the constants
fh1 = [1 -1];
[fh1t,fxh1,bh1,FtFh1] = preData(xhat, fh1, sizef);
FtF = FtF + FtFh1;

fv1 = [1 -1]';
[fv1t,fxv1,bv1,FtFv1] = preData(xhat, fv1, sizef);
FtF = FtF + FtFv1;

fh2 = [1 0 -1];
[fh2t,fxh2,bh2,FtFh2] = preData(xhat, fh2, sizef);
FtF = FtF + FtFh2;

fv2 = [1 0 -1]';
[fv2t,fxv2,bv2,FtFv2] = preData(xhat, fv2, sizef);
FtF = FtF + FtFv2;

% start the iteration
initer_max = 1;
outiter_max = 20;
for outiter = 1:outiter_max % Bregman iteration

    % solve for ( x^{(k+1),\{(b_j)^{(k+1)}\}} )
    for initer = 1:initer_max % quadratic splitting iteration
        % w sub-problem
        wh1 = wSubProb((fxh1+bh1)./alphas.h1,gamma*(alphas.h1)^2,betas.h1);
        wv1 = wSubProb((fxv1+bv1)./alphas.v1,gamma*(alphas.v1)^2,betas.v1);
        wh2 = wSubProb((fxh2+bh2)./alphas.h2,gamma*(alphas.h2)^2,betas.h2);
        wv2 = wSubProb((fxv2+bv2)./alphas.v2,gamma*(alphas.v2)^2,betas.v2);
        
        % x sub-problem
        ftws = zeros(size(Kty));
        ftwh1 = conv2(wh1.*alphas.h1 - bh1, fh1t, 'full');
        ftws = ftws + ftwh1;
        ftwv1 = conv2(wv1.*alphas.v1 - bv1, fv1t, 'full');
        ftws = ftws + ftwv1;
        ftwh2 = conv2(wh2.*alphas.h2 - bh2, fh2t, 'full');
        ftws = ftws + ftwh2;
        ftwv2 = conv2(wv2.*alphas.v2 - bv2, fv2t, 'full');
        ftws = ftws + ftwv2;
        num = lambda * (Kty + omega*fft2(y)) + gamma * fft2(ftws);
        denom = lambda * (KtK + omega*ItI) + gamma * FtF;
        Fx = num ./ denom;
        xhat = real(ifft2(Fx));
        
        % update the filter output
        fxh1 = conv2(xhat, fh1, 'valid');
        fxv1 = conv2(xhat, fv1, 'valid');
        fxh2 = conv2(xhat, fh2, 'valid');
        fxv2 = conv2(xhat, fv2, 'valid');
    end
    
    % update (b_j)^{(k+1)}, Bregman iteration
    bh1 = bh1 - wh1.*alphas.h1 + fxh1;
    bv1 = bv1 - wv1.*alphas.v1 + fxv1;
    bh2 = bh2 - wh2.*alphas.h2 + fxh2;
    bv2 = bv2 - wv2.*alphas.v2 + fxv2;
end

end

function [ft, fx, b, FtF] = preData(x, flit, sizef)

ft = rot90(flit,2);
fx = conv2(x, flit, 'valid');
b = zeros(size(fx));

FtF = abs(psf2otf(flit, sizef)).^2;

end

function [w] = wSubProb(fxb, gamma, beta)

if beta == 1
    w = max(abs(fxb) - 1 ./ gamma, 0) .* sign(fxb);
else
    w = wSubProb_solver(fxb, gamma, beta);
end

end

function [nI] = ditherMFI(I, winS)

hlfWS = floor(winS/2); % half block size
nI = I; % initilize
[~,nW] = size(nI); % image size
varMap = zeros(size(nI));
varMap(hlfWS+1:end-hlfWS,hlfWS+1:end-hlfWS) = reshape(var(im2col(nI,[winS winS])),size(nI)-[2*hlfWS 2*hlfWS]);

%%%%%%%% when there are three pixels in a row having the same value %%%%%%%

G = conv2(nI, [1 -1], 'same'); % d_{i,j} = x_{i,j+1} - x_{i,j}, horizontally
iZero = false(size(I));
iZero(:,2:end) = G(:,1:end-1) == 0 & G(:,2:end) == 0;
% if x_{i,j} is to be modified, then x_{i,j+1} cannot be modified, 
% otherwise it will influence the pixel value difference in modification
for iY = 1:nW-1 % horizontally, related the filter in use [1 -1]
    iZero_cur = iZero(:,iY);
    varMap_cur = varMap(:,iY);
    iZero_nex = iZero(:,iY+1);
    varMap_nex = varMap(:,iY+1);
    
    iZero_cur(iZero_cur & iZero_nex & varMap_cur <= varMap_nex) = false;
    iZero_nex(iZero_cur & iZero_nex & varMap_cur > varMap_nex) = false;
    
    iZero(:,iY) = iZero_cur;
    iZero(:,iY+1) = iZero_nex;
end

iMod = iZero;
nI(iMod) = nI(iMod) + sign(rand(sum(iMod(:)),1)-0.5);

%%%%%%%% when there are two pixels in a row having the same value %%%%%%%%%

G = conv2(nI, [1 -1], 'same'); % d_{i,j} = x_{i,j+1} - x_{i,j}, horizontally
iZero = G == 0;
% if x_{i,j} is to be modified, then x_{i,j+1} cannot be modified, 
% otherwise it will influence the pixel value difference in modification
for iY = 1:nW-1 % horizontally, related the filter in use [1 -1]
    iZero_cur = iZero(:,iY);
    varMap_cur = varMap(:,iY);
    iZero_nex = iZero(:,iY+1);
    varMap_nex = varMap(:,iY+1);
    
    iZero_cur(iZero_cur & iZero_nex & varMap_cur <= varMap_nex) = false;
    iZero_nex(iZero_cur & iZero_nex & varMap_cur > varMap_nex) = false;
    
    iZero(:,iY) = iZero_cur;
    iZero(:,iY+1) = iZero_nex;
end
varMap(~iZero) = -1; % for the sorting, remove the pixels which cannot be modified
[~,inds] = sort(varMap(:),'descend');
rate = 0.3; iMod = inds(1:ceil(rate*sum(iZero(:))));
nI(iMod) = nI(iMod) + sign(rand(numel(iMod),1)-0.5);

end