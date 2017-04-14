% Wrapper around kernelKNN
% kernelKNN_classify(K, trexs, testexs, y, k)
% K        kernel matrix
% trexs    training points index into K
% testexs  test points index into K
% y        labels 
% k        no. of nearest neighbors to use
function [acc, predLabels] = asymmetricKNN(Xtrain, Ytrain, Xtest, Ytest, PARAM)
					   
k = PARAM.k;

%% the kernelKNN routine takes in 5 arguments:
%% 1. the labels of the training points (given as Ytrain)

%% 2. a kernel matrix whose ij entry is the kernel fcn between test point i
%%    and training point j
Ktrain_test= formKernel(Xtrain, Xtest, PARAM);

%% 3. the self similarity of the training points (K_ii)
for i=1:size(Xtrain,1)
    nKtrain(i) = formKernel(Xtrain(i,:),Xtrain(i,:), PARAM);
end
%% 4. the self similarity of the test points
for i=1:size(Xtest,1)
    nKtest(i) = formKernel(Xtest(i,:),Xtest(i,:),PARAM);
end

if isfield(PARAM,'S') && ~isempty(PARAM.S)
    S = PARAM.S;
    Xlearn = PARAM.Xlearn;
    Ktest_learn  = formKernel(Xtest,  Xlearn, PARAM);
    Ktrain_learn = formKernel(Xtrain, Xlearn, PARAM);
    % Klearn_train or Ktrain_learn?
    Ktrain_test = Ktrain_test + Ktrain_learn*S*Ktest_learn';

    for i=1:size(Xtest,1)
	nKtest(i) = nKtest(i) + Ktest_learn(i,:)*S*Ktest_learn(i,:)';
    end
    for i=1:size(Xtrain,1)
	nKtrain(i) = nKtrain(i) + Ktrain_learn(i,:)*S*Ktrain_learn(i,:)';
    end
end

%% 5. the value of k  (given)
% call knn
predLabels = kernelKNN(Ytrain, Ktrain_test', nKtrain, nKtest, k)';

% compute accuracy
numRight = length(find(predLabels==Ytest));
acc  = numRight / length(predLabels); 


