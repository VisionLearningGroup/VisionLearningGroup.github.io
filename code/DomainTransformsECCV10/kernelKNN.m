% perform knn classification on nKtest
function [preds v] = kernelKNN(y, K, nKtr, nKtest, k)


add1 = 0;
if (min(y) == 0),
    y = y + 1;
    add1 = 1;
end
[n,ntr] = size(K);

D = zeros(n, ntr);
for (i=1:n),
    for (j=1:ntr),
        D(i,j) = nKtest(i) + nKtr(j) - 2 * K(i, j);
    end
end

[V, Inds] = sort(D');

preds = zeros(n,1);
for (i=1:n),
    counts = [];
    for (j=1:k),        
        if (y(Inds(j,i)) > length(counts)),
            counts(y(Inds(j,i))) = 1;
        else
            counts(y(Inds(j,i))) = counts(y(Inds(j,i))) + 1;
        end
    end
    [v(i), preds(i)] = max(counts);
end
if (add1 == 1),
    preds = preds - 1;
end