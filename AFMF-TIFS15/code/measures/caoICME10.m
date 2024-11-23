function [ f ] = caoICME10( I, d, tau )
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%CAOICME10 detects median filtering in digital images
% 
%   INPUT     I: image pixel matrix
%             d: square statistic region size
%           tau: threshold for building the variance matrix
%   OUTPUT
%             f: the forensic metric
% 
% reference:     G. Cao, Y. Zhao, R. Ni, L. Yu, and H. Tian
%                "Forensic Detection of Median Flitering in Digital
%                Images", ICME 2010
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
[nH,nW] = size(I); % image size
% The ceiling operation is according to Eq. (11) in the paper!
hlfWinS = ceil(d/2);

% the first order row-based difference binary map
% in both vertical and horizontal directions
forDMap_r = conv2(I,[1,-1],'same') == 0;
forDMap_r = forDMap_r(hlfWinS+1:end-hlfWinS,hlfWinS+1:end-hlfWinS);
forDMap_c = conv2(I,[1;-1],'same') == 0;
forDMap_c = forDMap_c(hlfWinS+1:end-hlfWinS,hlfWinS+1:end-hlfWinS);

% computing the variance map
Z = im2col(I,[2*hlfWinS+1 2*hlfWinS+1]); % all the over-lapping pathes
varMap = reshape(var(Z),nH-2*hlfWinS,nW-2*hlfWinS);
varMap = varMap >= tau;

% the frequency of zero values in the first-order difference of texture regions
f_r = sum(sum(forDMap_r.*varMap))./sum(varMap(:));
f_c = sum(sum(forDMap_c.*varMap))./sum(varMap(:));

% the forensic metric
f = f_r./sqrt(2) + f_c./sqrt(2);

end