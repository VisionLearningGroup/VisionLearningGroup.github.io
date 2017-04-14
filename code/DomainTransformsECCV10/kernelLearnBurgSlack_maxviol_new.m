function [K,iter,bhat] = kernelLearnBurgSlack_maxviol_new(C,gamma,K_0,tol, cycle_param)
% [A,iter, bhat] = metricLearnBurg(n,k,C,cycle_param,A_0,G_0,tol)
%
% Currently works only for constraints ||a_i1 - a_i2||^2 <= b
% Constraints are stored in a c x 5 matrix, each row of which
% contains x, i1, i2, b, and constraint type
% If the fifth column = 1, then the constraint is ||a_i1 - a_i2||^2 <= b
% If the fifth column = -1, then the constraint is ||a_i1 - a_i2||^2 >= b
%
% If cycle_param = 1, just does cyclic projection.
% If cycle_param = 2, gets most violated constraint every iteration.
%
% See also vNBregmanDC.m, test_gyrB_Burg.m.
%


[n,n] = size(K_0);
mxiter = 200000;
if (~exist('cycle_param')),
    cycle_param = 1;
end
if (~exist('tol')),
   tol = 1e-2; 
end
tol=1e-3;
%A = A_0;
K=K_0;
% % check to make sure all v's are non-zero
% valid = ones(size(C,1),1);
% for (i=1:size(C,1)),
%    i1 = C(i,2);
%    i2 = C(i,3);
%    v = G_0(i1,:)' - G_0(i2,:)'; 
%    if (norm(v) < 10e-10),
%       %disp('Warning: zero vector v found'); 
%       valid(i) = 0;
%    end
% end
% C = C(valid>0,:);

i = 1;
iter = 1;
[c,t] = size(C);
lambda = zeros(c,1);
runagain = 1;
bhat = C(:,4);
lambdaold = zeros(c,1);
%W1=zeros(n,n);
val = 1000;
mx=1000;
% V0=zeros(n,n,n);
% for i=1:n
%     for j=1:n
%         V0(:,i,j)=K_0(:,i)-K_0(:,j);
%     end
% end
while runagain == 1
    
    i1 = C(i,2);
    i2 = C(i,3);
    b = C(i,4);
    %v = G_0(i1,:)' - G_0(i2,:)';
    %w = B'*v;
    %wtw = v'*A*v;
    wtw=K(i1,i1)+K(i2,i2)-2*K(i1,i2);
    if (abs(bhat(i)) < 10e-10),
        error('bhat should never be 0!');
    end
    if C(i,5) == 1
        %alpha = min(lambda(i),.5*(1/(w'*w) - gamma/bhat(i)));
        %alpha = min(lambda(i),.5*(1/(wtw) - gamma/bhat(i)));
        alpha = min(lambda(i),(gamma/(gamma+1))*(1/(wtw) - 1/bhat(i)));
        lambda(i) = lambda(i) - alpha;
        %beta = alpha/(1 - alpha*w'*w);
        beta = alpha/(1 - alpha*wtw);        
        bhat(i) = gamma*bhat(i) / (gamma + alpha*bhat(i));
    elseif C(i,5) == -1
        %alpha = min(lambda(i),.5*(gamma/bhat(i) - 1/(w'*w)));
        %alpha = min(lambda(i),.5*(gamma/bhat(i) - 1/(wtw)));
        alpha = min(lambda(i),(gamma/(gamma+1))*(1/bhat(i) - 1/(wtw)));
        lambda(i) = lambda(i) - alpha;
        %beta = -1*alpha/(1 + alpha*w'*w);
        beta = -1*alpha/(1 + alpha*wtw);        
        bhat(i) = gamma*bhat(i) / (gamma - alpha*bhat(i));
    elseif C(i,5) == 0          
        %alpha = .5*(gamma/bhat(i) - 1/(wtw));
        alpha = (gamma / (gamma + 1))* (1/bhat(i) - 1/(wtw));
        lambda(i) = lambda(i) - alpha;
        %beta = -1*alpha/(1 + alpha*w'*w);
        beta = -1*alpha/(1 + alpha*wtw);        
        bhat(i) = gamma*bhat(i) / (gamma - alpha*bhat(i));
    end

    
    %[B, R] = cholupdatemult(B, beta, w);
%    A = A + (beta*A*v*v'*A);
    if(abs(beta)>1e-15)
        v=K(:,i1)-K(:,i2);
%           nu=eij+W1*v0;
%           W1=W1+beta*nu*nu';
          K=K+(beta*v*v');
    end
%    if beta < 0
%        R = cholupdate(eye(k),sqrt(abs(beta))*w,'-');
%    else
%        R = cholupdate(eye(k),sqrt(beta)*w,'+');
%    end
%    B = B*R';
    [i mx]=getmaxviol(K, C,bhat);  
    if(mod(iter,1000)==0)
        if(mx<tol || mxiter<iter)
            runagain=0;
        end
%         normsum = norm(lambda) + norm(lambdaold);
%         if (normsum == 0)
%             runagain = 0;
%         else
%             val = norm(lambdaold - lambda,1)/normsum;
%             if val < tol || iter > 20*size(C,1)
%                 runagain = 0;
%             end
%         end
%         lambdaold = lambda;
    end
    %i = mod(i,c) + 1;


    
    iter = iter + 1;
    if (mod(iter, 1000) == 0),  
       disp(sprintf('metric learn burg, iter: %d, conv = %f, mx = %f', iter, val, mx));
    end
end
disp(sprintf('metric learn burg, converged to tol: %f, iter: %d', val, iter));

