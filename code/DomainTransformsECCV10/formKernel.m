function K = formKernel(X1, X2, PARAM)

K = X1*X2';
if PARAM.use_Gaussian_kernel == 1
    %data is already normalized, choose arbitrary sigma
    K = exp(K-1);
else
    %this shouldn't actually do anything
    K = normalizedKernel(K);
end
