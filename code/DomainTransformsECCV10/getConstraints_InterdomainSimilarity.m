function [C,indices] = GetConstraints_InterdomainSimilarity(y1,y2,l,u)
pos=1;
ly1=length(y1);
ly2=length(y2);
C=zeros(ly1*ly2,4);
for i=1:ly1
    for j=1:ly2
        if(y1(i)==y2(j))
            C(pos,:)=[i j+ly1 u -1];
        else
            C(pos,:)=[i j+ly1 l 1];
        end
        pos=pos+1;
    end
end
m=ly1+ly2;
indices=[1:m];
%C = C(rp,:);