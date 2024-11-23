function [ rho ] = kirchnerSPIE10( I, k, l, blkSize )
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%KIRCHNERSPIE10 detects median filtering in digital images
% 
%   INPUT     I: image pixel matrix
%         (k,l): lag for computing the first-order difference image
%       blkSize: indicating the block size, if the forensic metric is
%                computed by taking the weighted median of the simple
%                forensic metric
%   OUTPUT
%           rho: the forensic metric
% 
% reference:     M. Kirchner, and J. Fridrich
%                "On Detection of Median Filtering in Digital Images", 
%                SPIE 2010
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
% last modified: Oct. 15th, 2013
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

I = double(uint8(I)); % change the pixel values into float number
D_kl = I(1:end-k,1:end-l) - I(k+1:end,l+1:end); % first-order difference image

if nargin >= 4
    if k > 0
        % remove the difference across the block borders (vertically)
        rows = []; offset = 1;
        while blkSize-k+offset <= blkSize
            rows = cat(2, rows, (blkSize-k+offset):blkSize:size(D_kl,1));
            offset = offset + 1;
        end
        D_kl(rows,:) = [];
    end
    if l > 0
        % remove the difference across the block borders (horizontally)
        cols = []; offset = 1;
        while blkSize-l+offset <= blkSize
            cols = cat(2, cols, (blkSize-l+offset):blkSize:size(D_kl,2));
            offset = offset + 1;
        end
        D_kl(:,cols) = [];
    end
    
    bD_kl = im2vec(D_kl,[blkSize-k,blkSize-l],0);
    rho_b = histc(bD_kl,0)./histc(bD_kl,1);
    w_b = 1 - histc(bD_kl,0)./(blkSize.^2-blkSize); % weight
    rho = median(w_b(~isnan(rho_b) & ~isinf(rho_b)).*rho_b(~isnan(rho_b) & ~isinf(rho_b))); % some rho_b may be NAN when h1 == 0
else
    rho = histc(D_kl(:),0)./histc(D_kl(:),1);
end

end

