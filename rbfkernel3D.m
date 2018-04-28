function [K]=rbfkernel3D(X1, X2, size)
    [dim1,~] = size(X1);
    [dim2,~] = size(X2);
    num1 = dim1/size;
    num2 = dim2/size;
    X1_rep = repmat(X1, 1, num2);
    X2_rep = repmat(X2, 1, num1);
    
end