function [K, S] = kernelMetricLearning_maxviol(K0, C, gamma, thresh, targets)

if (~exist('thresh')),
    thresh = 10e-3;
end

if (~exist('gamma')),
    gamma = 1;
end

[n,n] = size(K0);
num_constraints = size(C, 1);
Cburg = zeros(num_constraints, 5);
Cburg(:, 1) = zeros(num_constraints, 1);
Cburg(:, 2:3) = C(:,1:2);
Cburg(:, 5) = C(:,3);

if (~exist('targets')),
    [l, u] = computeKernelDistanceExtremes(K0, 5, 95);
    % assign lte constraints to have r.h.s. of l
    Cburg(find(C(:,3)>0), 4) = l;
    % assign gte constraints to have r.h.s. of u
    Cburg(find(C(:,3)<0), 4) = u;
else
    Cburg(:,4) = targets;
end
S=1;
try    
     K= kernelLearnBurgSlack_maxviol_new(Cburg, gamma, K0, thresh);
catch   
    disp('kernelMetricLearning_maxviol: Unable to learn mahal matrix');
    le = lasterror;
    disp(le.message);
    K = eye(n,n);
end    
