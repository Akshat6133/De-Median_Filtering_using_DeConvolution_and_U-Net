function [w] = wSubProb_solver(v, gamma, beta)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function is the solve_image_bregman.m function (yet with some 
% modifications) included in the Matlab code package of their CVPR 2011 
% paper "Blind Deconvolution using a Normalized Sparsity Measure", by 
% Dilip Krishnan, Terence Tay and Rob Fergus.
% 
% modified by: Wei FAN (wei.fan@gipsa-lab.grenoble-inp.fr)
% 
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% solve the following component-wise separable problem
% min maskk .* |w|^\beta + \frac{\gamma}{2} (w - v).^2
% 
% range of input data and step size; increasing the range of decreasing
% the step size will increase accuracy
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

range = 10;
step  = 0.001;

xx = -range:step:range;
tmp = compute_w(xx, gamma, beta);
w = interp1(xx', tmp(:), v(:), 'linear', 'extrap');
w = reshape(w, size(v,1), size(v,2));

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% call different functions to solve the minimization problem
% min |w|^\beta + \frac{\gamma}{2} (w - v).^2 for a fixed beta and alpha
%
function w = compute_w(v, gamma, beta)

if (abs(beta - 1) < 1e-9)
    % assume beta = 1.0
    w = compute_w1(v, gamma);
    return;
end

if (abs(beta - 2/3) < 1e-9)
    % assume beta = 2/3
    w = compute_w23(v, gamma);
    return;
end

if (abs(beta - 1/2) < 1e-9)
    % assume beta = 1/2
    w = compute_w12(v, gamma);
    return;
end

% for any other value of beta, plug in some other generic root-finder
% here, we use Newton-Raphson
w = newton_w(v, gamma, beta);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function w = compute_w23(v, gamma)
% solve a quartic equation
% for beta = 2/3

epsilon = 1e-6; %% tolerance on imag part of real root

k = 8/(27*gamma^3);
m = ones(size(v))*k;

% Now use formula from
% http://en.wikipedia.org/wiki/Quartic_equation (Ferrari's method)
% running our coefficients through Mathmetica (quartic_solution.nb)
% optimized to use as few operations as possible...

%%% precompute certain terms
v2 = v .* v;
v3 = v2 .* v;
v4 = v3 .* v;
m2 = m .* m;
m3 = m2 .* m;

%% Compute alpha & beta
alpha = -1.125*v2;
beta2 = 0.25*v3;

%%% Compute p,q,r and u directly.
q = -0.125*(m.*v2);
r1 = -q/2 + sqrt(-m3/27 + (m2.*v4)/256);

u = exp(log(r1)/3);
y = 2*(-5/18*alpha + u + (m./(3*u)));

W = sqrt(alpha./3 + y);

%%% now form all 4 roots
root = zeros(size(v,1),size(v,2),4);
root(:,:,1) = 0.75.*v  +  0.5.*(W + sqrt(-(alpha + y + beta2./W )));
root(:,:,2) = 0.75.*v  +  0.5.*(W - sqrt(-(alpha + y + beta2./W )));
root(:,:,3) = 0.75.*v  +  0.5.*(-W + sqrt(-(alpha + y - beta2./W )));
root(:,:,4) = 0.75.*v  +  0.5.*(-W - sqrt(-(alpha + y - beta2./W )));

%%%%%% Now pick the correct root, including zero option.

%%% Clever fast approach that avoids lookups
v2 = repmat(v,[1 1 4]);
sv2 = sign(v2);
rsv2 = real(root).*sv2;

%%% condensed fast version
%%%             take out imaginary                roots above v/2            but below v
root_flag3 = sort(((abs(imag(root))<epsilon) & ((rsv2)>(abs(v2)/2)) &  ((rsv2)<(abs(v2)))).*rsv2,3,'descend').*sv2;
%%% take best
w = root_flag3(:,:,1);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function w = compute_w12(v, gamma)
% solve a cubic equation
% for beta = 1/2

epsilon = 1e-6; %% tolerance on imag part of real root

k = -0.25/gamma^2;
m = ones(size(v))*k.*sign(v);

%%%%%%%%%%%%%%%%%%%%%%%%%%% Compute the roots (all 3)
t1 = (2/3)*v;

v2 = v .* v;
v3 = v2 .* v;

%%% slow (50% of time), not clear how to speed up...
t2 = exp(log(-27*m - 2*v3 + (3*sqrt(3))*sqrt(27*m.^2 + 4*m.*v3))/3);

t3 = v2./t2;

%%% find all 3 roots
root = zeros(size(v,1),size(v,2),3);
root(:,:,1) = t1 + (2^(1/3))/3*t3 + (t2/(3*2^(1/3)));
root(:,:,2) = t1 - ((1+1i*sqrt(3))/(3*2^(2/3)))*t3 - ((1-1i*sqrt(3))/(6*2^(1/3)))*t2;
root(:,:,3) = t1 - ((1-1i*sqrt(3))/(3*2^(2/3)))*t3 - ((1+1i*sqrt(3))/(6*2^(1/3)))*t2;

root(isnan(root) | isinf(root)) = 0; %%% catch 0/0 case

%%%%%%%%%%%%%%%%%%%%%%%%%%%% Pick the right root
%%% Clever fast approach that avoids lookups
v2 = repmat(v,[1 1 3]);
sv2 = sign(v2);
rsv2 = real(root).*sv2;
root_flag3 = sort(((abs(imag(root))<epsilon) & ((rsv2)>(2*abs(v2)/3)) &  ((rsv2)<(abs(v2)))).*rsv2,3,'descend').*sv2;
%%% take best
w = root_flag3(:,:,1);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function w = compute_w1(v, gamma)
% solve a simple max problem for beta = 1

w = max(abs(v) - 1/gamma, 0).*sign(v);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function w = newton_w(v, gamma, beta)

% for a general alpha, use Newton-Raphson; more accurate root-finders may
% be substituted here; we are finding the roots of the equation:
% % \beta*|w|^{\beta - 1} + \gamma*(v - w) = 0
% \beta*|w|^{\beta - 1}sgn(w) + \gamma*(w - v) = 0

iterations = 4;

x = v;

for a = 1:iterations
    fd = (beta)*sign(x).*abs(x).^(beta-1)+gamma*(x-v);
    fdd = beta*(beta-1)*abs(x).^(beta-2)+gamma;
    
    x = x - fd./fdd;
end

x(isnan(x)) = 0;

% check whether the zero solution is the better one
z = gamma/2*v.^2;
f = abs(x).^beta + gamma/2*(x-v).^2;
w = (f<z).*x;

end