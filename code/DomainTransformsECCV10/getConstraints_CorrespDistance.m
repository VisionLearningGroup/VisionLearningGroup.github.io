% Matches is an n-by-2 array, where Matches(i,1:2) are indices into the
% source and target examples that are a match in terms of object id and
% viewpoint.
function [C,indices] = GetConstraints_CorrespDistance(y1,y2,Matches)

%disp('USING DISTANCE CORRESPONDENCE CONSTRAINTS')
%%
%% generate SMALL DISTANCE constraints based on GT matches    
%%
ly1=length(y1);
ly2=length(y2);
Cnear=zeros(size(Matches,1),3);
for k=1:size(Matches,1)
    Cnear(k,:)=[Matches(k,1) Matches(k,2)+ly1 +1];
end

%% Generate class-based LARGE DISTANCE constraints 
pos=1;
Cfar=zeros(ly1*ly2,3);
for i=1:ly1
    for j=1:ly2
        % add the pair if they have diff. labels
        if(y1(i)~=y2(j))
            Cfar(pos,:)=[i j+ly1 -1]; 
            pos=pos+1;
        end
    end
end
Cfar(pos:end,:) = [];
C = [Cnear; Cfar];

m=ly1+ly2;
indices=[1:m];




%% set number of negative constraints
%%num_neg_constraints = size(Cpos,1);    % same as no. of positive constr.
% num_neg_constraints = size(Cneg,1);    % keep all
% rind = randperm(size(Cneg,1));
% Cneg = Cneg(rind(1:num_neg_constraints),:);


