function K = NDrbfkernel(XM,XN,theta)
    [lenM, dim] = size(XM);
    lenN = size(XN,1);
    XMrep = repmat(XM,1,lenN);
    XNrep = repmat(reshape(XN', 1, dim*lenN), lenM, 1);
    diffplane = -theta(2)*(XMrep - XNrep).^2;
    diffcube = sum(reshape(diffplane, lenM, dim, lenN),2);
    K = zeros(lenM, lenN);
    K(:, :) = theta(1)*exp(diffcube);
end