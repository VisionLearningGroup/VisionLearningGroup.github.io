% Matches is an n-by-2 array, where Matches(i,1:2) are indices into the
% source and target examples that are a match in terms of object id and
% viewpoint.
function [C,indices] = GetConstraints_CorrespSimilarity(y1,y2,Matches,l,u)

%disp('USING SIMILARITY CORRESPONDENCE CONSTRAINTS')
%%
%% generate SIMILARITY constraints based on GT matches    
%%
ly1=length(y1);
ly2=length(y2);
Csim=zeros(size(Matches,1),4);
for k=1:size(Matches,1)
    Csim(k,:)=[Matches(k,1) Matches(k,2)+ly1 u -1];
end

%% Generate class-based DISSIMILARITY constraints 
pos=1;
Cdis=zeros(ly1*ly2,4);
for i=1:ly1
    for j=1:ly2
        % add the pair if they have diff. labels
        if(y1(i)~=y2(j))
            Cdis(pos,:)=[i j+ly1 l 1]; 
            pos=pos+1;
        end
    end
end
Cdis(pos:end,:) = [];
C = [Csim; Cdis];

m=ly1+ly2;
indices=[1:m];




%% set number of negative constraints
%%num_neg_constraints = size(Cpos,1);    % same as no. of positive constr.
% num_neg_constraints = size(Cneg,1);    % keep all
% rind = randperm(size(Cneg,1));
% Cneg = Cneg(rind(1:num_neg_constraints),:);


