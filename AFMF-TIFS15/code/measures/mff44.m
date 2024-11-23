function [ F ] = mff44( IMG, iScal )
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%MFF44 extracts 44-D MFF features (or the scalar) for median filtering detection
% 
%   INPUT   IMG: image name / image pixel value matrix
%         iScal: output a merged feature scalar or a vector
%   OUTPUT
%             F: 44-D MFF features (default)
%                output the merged discriminating feature (scalar) when
%                iScal is indicated to be true
% 
% reference:     H.-D. Yuan,
%                "Blind Forensics of Median Filtering in Digital Images", 
%                TIFS 2011
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
% last modified: Nov. 1st, 2013
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if ischar(IMG)
    I = double(imread(IMG)); % read image file
else
    I = double(uint8(IMG)); % image pixel value matrix
end
[nH,nW] = size(I); % image size

I = I(1:floor(nH/3)*3,1:floor(nW/3)*3); % proper cropping
X = im2vec(I,3,0); % split into non-overlapping 3*3 blocks
X_c = X(5,:); % the center pixel value of the block
X_ordered = sort(X); % sorting
X_m = X_ordered(5,:); % the median of the block

% The Distribution of the Block Median (DBM)
h_DBM = sum(X == repmat(X_m,[9,1]),2)./size(X,2);

% The Occurrence of the Block-Center Gray Level (OBC)
h_OBC = histc(sum(X == repmat(X_c,[9,1])),1:9)'./size(X,2);

% The Quantity of Gray Levels in a Block (QGL)
tp = ones(1,size(X_ordered,2));
for k = 2:9
    tp = tp + (X_ordered(k,:)~=X_ordered(k-1,:));
end
h_QGL = histc(tp,1:9)'./size(X,2);

% The Distribution of the Block-Center Gray Level in the Sorted Gray Levels (DBC)
h_DBC = sum(X_ordered == repmat(X_c,[9,1]),2)./size(X,2);

% The First Occurrence of the Block-Center Gray Level in the Sorted Gray Levels (FBC)
tp = X_ordered == repmat(X_c,[9,1]);
h_FBC = zeros(9,1);
for k = 1:9
    h_FBC(k) = sum(tp(k,:));
    tp(k+1:end,tp(k,:)==1) = 0;
end
h_FBC = h_FBC./size(X,2);

if nargin >= 2 && iScal
    % The merged discriminating feature (scalar)
    Fnumerator = h_DBM(5)*h_OBC(2)*h_QGL(6)*(h_DBC(3)+h_DBC(7)-h_DBC(2)-h_DBC(8))*h_FBC(3);
    Fdenominator = h_OBC(1)*h_QGL(9)*(h_DBC(2)+h_DBC(8)-h_DBC(1)-h_DBC(9))*h_FBC(2)*h_FBC(9);
    F = Fnumerator/Fdenominator;
else    
    % The MFF feature vector
    F = [h_DBM(1:4);h_DBM(6:end); h_OBC; h_QGL; h_DBC; h_FBC];
end

end

