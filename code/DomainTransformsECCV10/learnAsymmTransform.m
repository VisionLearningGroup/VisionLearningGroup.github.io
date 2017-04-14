% params = learnAsymmTransform(XA, yA, XB, yB, params)
%
% Learns a transformation between a pair of domains A and B,
% using the ARC-t algorithm published in [Kulis et al. CVPR11]
%
% XA, yA: examples and labels in source domain
% XB, yB: examples and labels in target domain
% params.constraint_type: 'allpairs' 'interdomain'
% params.constraint_num:  number of constraints to use
% params.gamma:  gamma parameter 
% params.use_Gaussian_kernel: 0 (linear) or 1 (Gaussian)
%
% Returns: 
% params.S: learned kernel parameters
% params.Xlearn: training points

function params = learnAsymmTransform(XA, yA, XB, yB, params)

% concatenate training examples into one matrix
X = [XA; XB];
y = [yA yB];
trainlabels_ml = [yA yB];

%form the kernel matrix over the training data 
K0train = formKernel(X, X, params);

% form constraints between inter-domain points
[l u] = getKernelValueExtremes(K0train, .02, .98);

if strcmp( params.constraint_type, 'interdomain')
    % generate constr. between inter-domain points only
    [C indices] = getConstraints_InterdomainSimilarity(yA,yB,l,u);
    
elseif strcmp( params.constraint_type, 'corresp')
    [C,indices] = getConstraints_CorrespSimilarity(yA,yB,params.MatchesTrain,l,u);
    
else
    error('Unsupported constraint type');
end

fprintf('Learning asymmetric (Frobenius norm) transform\n');
[S slack] = asymmetricFrob_slack_kernel(K0train,C,params.gamma,10e-3);

% post-processing so we can compute kernel for new points
params.S = S;
params.Xlearn = X(indices,:);

% Note, final kernel for new data X is K= X*L*X' : 
% X*L*X' = 
% X*( eye(d) + X(indices,:)'*S*X(indices,:))*X' = 
% X*X' + X*X(indices,:)'*S*X(indices,:)*X' = 
% K0+K0(:,indices)*S*K0(:,indices)' =
% K
