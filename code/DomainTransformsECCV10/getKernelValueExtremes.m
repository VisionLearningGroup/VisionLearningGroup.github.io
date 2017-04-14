
function [l u] = getKernelValueExtremes(K, lowerPct, upperPct)

[v1,v2] = sort(K(:),'ascend');

% TODO: pass percentile as params
l = v1(ceil(lowerPct*length(v1)));
u = v1(ceil(upperPct*length(v1)));
