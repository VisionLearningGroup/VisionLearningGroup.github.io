function [S slack] = AsymmetricFrob_slack_kernel(K0,C,gamma,thresh)
if (~exist('thresh')),
    thresh=10e-3;
end
if (~exist('gamma')),
    gamma = 1e1;
end
maxit = 1e6;
[n,n] = size(K0);
S = zeros(n,n);
[c,t] = size(C);
slack = zeros(c,1);
lambda = zeros(c,1);
lambda2 = zeros(c,1);
v = (C(:,1)-1)*n+C(:,2);
viol = C(:,4).*(K0(v)-C(:,3));
viol = viol';
%for i = 1:c
%    if mod(i,1000) == 0
%        i
%    end
%    %kx = K0(C(i,1),:);
%    %ky = K0(:,C(i,2));
%    viol(i) = C(i,4)*(K0(C(i,1),C(i,2)) - C(i,3));
%end
for i = 1:maxit
    [mx,curri] = max(viol);
    if mod(i,1000) == 0
        disp(sprintf('Iteration %d, maxviol %d', i, mx));
        if mx < thresh
            break;
        end
    end
    p1 = C(curri,1);
    p2 = C(curri,2);
    b = C(curri,3);
    s = C(curri,4);
    kx = K0(p1,:);
    ky = K0(:,p2);
    
    alpha = min(lambda(curri),(s*(b-K0(p1,p2)-kx*S*ky)-slack(curri)) / (1/gamma + K0(p1,p1)*K0(p2,p2)));
    lambda(curri) = lambda(curri) - alpha;
    S(p1,p2) = S(p1,p2) + s*alpha;
    slack(curri) = slack(curri) - alpha/gamma;
    alpha2 =  min(lambda2(curri),gamma*slack(curri));
    slack(curri) = slack(curri) - alpha2/gamma;
    lambda2(curri) = lambda2(curri) - alpha2;
        
    %update viols
    v = K0(C(:,1),p1);
    w = K0(p2,C(:,2))';
    viol = viol + s*alpha*C(:,4)'.*K0(C(:,1),p1)'.*K0(p2,C(:,2));
    viol(curri) = viol(curri) + (alpha+alpha2)/gamma;
end
%check viols
%for i = 1:50
%    violcheck(i) = C(i,4)*(K0(C(i,1),C(i,2)) + K0(C(i,1),:)*S*K0(:,C(i,2)) - C(i,3)) - slack(i);
%end
%[viol(1:50)' violcheck']
