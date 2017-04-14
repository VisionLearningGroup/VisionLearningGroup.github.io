%function [trexsA, testexsA, trexsB, testexsB] = splitAllCategories(yA, yB, Objs, PARAM)
%
% Split all categories into train and test, in domains A and B
%
% Input:
% yA, yB: labels of data in domains A and B
% Objs: cell array of object ids in A, B (webcam and dslr domains only)
% PARAM: parameters set by config file, e.g. config_diffcat_dslr_webcam.m
%
% Returns:
% trexsA, trexsB:  indices of a random subset of examples in A and B, for 
%                  training domain transform.
% testexsA,testexsB: indices into random subset of examples in  A and B, 
%                    for training and testing k-nearest neighbors object 
%                    classification.
% validationA, validationB: validation examples, if requested in PARAM
%
% Note that in webcam/dslr there are typically 5 object ids per category.
% For some experiments we may want to hold out whole objects: 
% i.e. if we train on obj ids=1,2,3, then we want to test only on 4,5.
% This is specified by PARAM.[test|train]IDs_[A|B] variables.

function [trexsA, testexsA, trexsB, testexsB] = splitAllCategories(yA, yB, Objs, PARAM)

% total number of categories
numclasses = length(PARAM.categories);

%number of labeled training samples in each domain to use
num_training_A = PARAM.num_training_A;
num_training_B = PARAM.num_training_B;

nA = length(yA);
nB = length(yB);

trexsA=[]; testexsA=[]; trexsB=[]; testexsB=[];

% choose train and test images for domain B and A
% there are typically 5 object ids per category
% For dslr/webcam combination, we hold out whole objects: 
% i.e. if we train on obj ids=1,2,3, then we want to test on 4,5 
% ECCV10 PAPER: trained on obj1-3 for webcam, obj1 for dslr
for i = 1:numclasses
    %%%%%%% domain A %%%%%%%
    indA = find(yA==i);
    if isempty(Objs{1})  % domain does not have object ids
        rpA=randperm(length(indA));
        trexsA = [trexsA indA(rpA(1:num_training_A))];
        testexsA = [testexsA indA(rpA(num_training_A+1:end))];
    else
        tmp=Objs{1}(indA);  % object ids of images in this category
        % choose a random subset of images from train objects
        objA = indA( find(ismember(tmp, PARAM.trainIDs_A)) );
        rindx = randperm(length(objA));
        trexsA=[trexsA objA(rindx(1:min(length(objA),num_training_A)))];
        testexsA=[testexsA indA( find(ismember(tmp, PARAM.testIDs_A)) )];
    end
    %%%%%%% domain B %%%%%%%
    indB = find(yB==i);
    if isempty(Objs{2})  % domain does not have object ids
        rpB=randperm(length(indB));
        trexsB = [trexsB indB(rpB(1:num_training_B))];
        testexsB = [testexsB indB(rpB(num_training_B+1:end))];
    else
        tmp=Objs{2}(indB);  % object ids of images in this category
        % limit the number of images of train objs, pick at random
        objB = indB( find(ismember(tmp, PARAM.trainIDs_B)) );
        rindx = randperm(length(objB));
        trexsB=[trexsB objB(rindx(1:min(length(objB),num_training_B)))];
        testexsB=[testexsB indB( find(ismember(tmp,  PARAM.testIDs_B)) )];
    end
end
