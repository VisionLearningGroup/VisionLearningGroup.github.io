%function [trexsA, testexsA, trexsB, testexsB, validationA, validationB] = splitDiffCategories(yA, yB, Objs, PARAM)
%
% Sample training categories in domains A and B to get training data for 
% learning transform; split test categories into train/test data for 
% object classification.
%
% Input:
% yA, yB: labels of data in domains A and B
% Objs: cell array of object ids in A, B (webcam and dslr domains only)
% PARAM: parameters set by config file, e.g. config_diffcat_dslr_webcam.m
%
% Returns:
% trexsA, trexsB:  indices of a random subset of examples of training 
%                  classes in A and B, for training transform
% testexsA,testexsB: indices into random subset of examples of test
%                    classes, for training and testing object
%                    classification.
% validationA, validationB: validation examples, if requested in PARAM
%
% Note that in webcam/dslr there are typically 5 object ids per category.
% For some experiments we may want to hold out whole objects: 
% i.e. if we train on obj ids=1,2,3, then we want to test only on 4,5.
% This is specified by PARAM.[test|train]IDs_[A|B] variables.

function [trexsA, testexsA, trexsB, testexsB, validationA, validationB] = splitDiffCategories(yA, yB, Objs, PARAM)

% total number of train and test categories
trainClasses = PARAM.classes_train;
testClasses = PARAM.classes_test;
numTrainClasses = length(trainClasses);
numTestClasses = length(testClasses);


%number of labeled training samples in each domain to use
num_training_A = PARAM.num_training_A;
num_training_B = PARAM.num_training_B;
num_testing_A = PARAM.num_testing_A;
num_testing_B = PARAM.num_testing_B;

nA = length(yA);
nB = length(yB);

trexsA=[]; testexsA=[]; trexsB=[]; testexsB=[]; validationA = []; validationB =[]; 

% ECCV PAPER: trained on obj1-3 for webcam, obj1 for dslr (only one)
for i = 1:numTrainClasses
    current_class = trainClasses(i);
    
    %%%%%%% domain A Training %%%%%%%
    indA = find(yA==current_class);
    
    if isempty(Objs{1})  %amazon does not have object ids
        rpA=randperm(length(indA));
        trexsA = [trexsA indA(rpA(1:num_training_A))];
    else
        tmp=Objs{1}(indA);  % object ids of images in this category
        % choose a random subset of images from train objects
        objA = indA( find(ismember(tmp, PARAM.trainIDs_A)) );
        rindx = randperm(length(objA));
        trexsA=[trexsA objA(rindx(1:min(length(objA),num_training_A)))];
    end
    
    %%%%%%% domain B Training %%%%%%%
    % todo: handle the case when domain B does not have object ids
    indB = find(yB==current_class);
    tmp=Objs{2}(indB);
    % limit the number of images of train objs, pick at random
    objB = indB( find(ismember(tmp, PARAM.trainIDs_B)) );
    rindx = randperm(length(objB));
    trexsB=[trexsB objB(rindx(1:min(length(objB),num_training_B)))];
    
end

for i = 1:numTestClasses
    current_class = testClasses(i);
    indA = find(yA==current_class);

    tmp=Objs{1}(indA);  % object ids of images in this category
    % choose a random subset of images from train objects
    objA = indA( find(ismember(tmp, PARAM.testIDs_A)) );
    rindx = randperm(length(objA));
    testexsA=[testexsA objA(rindx(1:min(length(objA),num_testing_A)))];
   
    validationA = [validationA indA(find(ismember(tmp, PARAM.validationIDs_A))) ];


    %%%%%%% domain B Training %%%%%%%
    indB = find(yB==current_class);
    
    tmp=Objs{2}(indB);    
    % limit the number of images of train objs, pick at random
    objB = indB( find(ismember(tmp, PARAM.testIDs_B)) );
    rindx = randperm(length(objB));
    testexsB=[testexsB objB(rindx(1:min(length(objB),num_testing_B)))];
    
    validationB = [validationB indB( find(ismember(tmp, PARAM.validationIDs_B)) ) ];
    
end

