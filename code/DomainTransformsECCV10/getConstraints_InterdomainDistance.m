function [C,indices] = GetConstraints_InterdomainDistance(y1,y2)
pos=1;
ly1=length(y1);
ly2=length(y2);
C=zeros(ly1*ly2,3);
for i=1:ly1
    for j=1:ly2
        if(y1(i)==y2(j))
            C(pos,:)=[i j+ly1 1];
        else
            C(pos,:)=[i j+ly1 -1];
        end
        pos=pos+1;
    end
end
m=ly1+ly2;
indices=[1:m];
%C = C(rp,:);