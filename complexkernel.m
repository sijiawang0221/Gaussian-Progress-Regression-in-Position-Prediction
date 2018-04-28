function K = complexkernel(XM, XN, theta_c)
    % XM: m x d matrix
    % XN: n x d matrix
    % theta_c: include hyperparameters theta and 
    %          kernel value c, like [theta, c]
    
    % informative features
    XMc = XM(:, 2:end);
    XNc = XN(:, 2:end);
    K2 = theta_c(1)*exp(-theta_c(2)*pdist2(XMc, XNc).^2);
    
    % ID feature
    IDM = XM(:, 1);
    IDN = XN(:, 1);
    K1 = repmat(IDM, 1, length(IDN)) == repmat(IDN', length(IDM), 1);
    K1(K1 ~= 1) = theta_c(3);
    
    % final result
    K = K1.*K2;
end