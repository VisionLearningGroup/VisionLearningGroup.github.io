function [l, u] = ComputeKernelDistanceExtremes(K0, a, b, R)
% function [l, u] = ComputeDistanceExtremes(X, a, b)
% compute lower bound Euclidean distance at percentile a
% compute upper bound Euclidean distance at percentile b
% a and b must be integers between 1 and 100

if (a < 1 || a > 100),
    error('a must be between 1 and 100')
end
if (b < 1 || b > 100),
    error('b must be between 1 and 100')
end


[n, n] = size(K0);

num_trials = min(100, n*(n-1)/2);

% we will sample with replacement
dists = zeros(num_trials, 1);
for (i=1:num_trials),
    j1 = ceil(rand(1)*n);
    j2 = ceil(rand(1)*n);    
    dists(i) = K0(j1,j1)+K0(j2,j2)-2*K0(j1,j2);
end


[f, c] = hist(dists, 100);
l = c(floor(a));
u = c(floor(b));