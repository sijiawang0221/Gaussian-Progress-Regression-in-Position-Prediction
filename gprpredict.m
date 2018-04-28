function [posmean, poscov] = gprpredict(xtr, ytr, xte, mu, sigma)
    nE=diag(ones(length(xtr),1)*sigma);
    close all
    kernel = @(XN,XM, theta) theta(1)*exp(-theta(2)*pdist2(XN,XM).^2);
    Ker = kernel(xte,xtr,[1 0.5]);
    Krr = kernel(xtr,xtr,[1 0.5]);
    Kee = kernel(xte,xte,[1 0.5]);
    posmean = mu + Ker/(Krr+nE)*(ytr - mean(xtr)');
    poscov = Kee-Ker*(Krr+nE)*Ker';
end