function [xtr, ytr, xte, yte] = prefetch(data)
    trsize = floor(0.8*size(data,1));
    xtr = data(1:trsize,1:end-1);
    ytr = data(1:trsize,end);
    xte = data(trsize+1:end,1:end-1);
    yte = data(trsize+1:end,end);
end