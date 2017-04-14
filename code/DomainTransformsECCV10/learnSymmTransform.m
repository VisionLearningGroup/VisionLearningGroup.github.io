% params = learnAsymmTransform(XA, yA, XB, yB, params)
%
% Learns a transformation between a pair of domains A and B,
% using the algorithm published in [Saenko et al. ECCV10]
%
% XA, yA: examples and labels in source domain
% XB, yB: examples and labels in target domain
% params.constraint_type: 'allpairs' 'interdomain' or 'corresp'
% params.gamma:  gamma parameter (tradeoff constraint satisfaction)
% params.use_Gaussian_kernel: 0 (linear) or 1 (Gaussian)
% params.constraint_num:  (optional) number of constraints to use
% 
% Returns: 
% params.S: learned kernel parameters
% params.Xlearn: training points
%
function params = learnSymmTransform(XA, yA, XB, yB, params)

% concatenate training examples into one matrix
X = [XA; XB];
y = [yA yB];
trainlabels_ml = [yA yB];

%form the kernel matrix over the training data 
K0train = formKernel(X, X, params);

% how many constraints to use?
num_constraints = length(trainlabels_ml)^2;
if isfield(params, 'constraint_num') && params.constraint_num>0
    num_constraints = params.constraint_num;
end

% Generate constraints based on type
if strcmp( params.constraint_type, 'allpairs')
    % generate constraints over all (or subset of) pairs of training points
    [C indices] = getConstraints_AllpairsDistance(trainlabels_ml, num_constraints);
    
elseif strcmp( params.constraint_type, 'interdomain')
    % generate constr. between inter-domain points only
    [C indices] = getConstraints_InterdomainDistance(yA,yB);

elseif strcmp( params.constraint_type, 'corresp')
    [C indices] = getConstraints_CorrespDistance(yA,yB,params.MatchesTrain);
else
    error('Unknown constraint type');
end

%get kernel matrix over constraint points
K0const=K0train(indices,indices);

fprintf('Learning symmetric transform\n');
Kconst=kernelMetricLearning_maxviol(K0const, C, params.gamma);

%this is postprocessing so we can compute the kernel values over all points
iK = pinv(K0const);
S = iK*(Kconst - K0const)*iK;

params.S = S;
params.Xlearn = X(indices,:);
