function [C,indices] = GetConstraints_AllpairsDistance(y, num_constraints)
%choose a set of points and then constraint each point with another
num_points=floor(sqrt(num_constraints));
m=length(y);
rp=randperm(m);
indices=rp(1:num_points);
z=y(indices);
pos=1;
C=zeros((num_points*(num_points-1))/2,3);
for i=1:num_points
    for j=i+1:num_points
        if(z(i)==z(j))
            C(pos,:)=[i j 1];
        else
            C(pos,:)=[i j -1];
        end
        pos=pos+1;
    end
end