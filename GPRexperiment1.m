function [ypred, RMSE]=GPRexperiment1(GPRmdls,xVal,yVal,xTe,yTe)

% training phase: 
% using 60% of data as training set (size of N), train N GPRs' function (hi, i=1~N)
% mdls = {N}


% validation phase:
% using 20% of data as validtion set (size of M)
% using SVM to learning weights, H=[hj(x_val_i)] M by N matrix as inputs, validation set as 
N = length(GPRmdls);
M = length(yVal);
K = length(yTe);
% creating H matrix
H=zeros(M,N);
parfor i=1:N
    H(:,i)=predict(GPRmdls{i},xVal);
end
% Now, problem becomes learning w for min[H(x')*w] using given xVal
SVMmdl = fitrsvm(H,yVal,'KernelFunction','gaussian','KernelScale',...
            'auto','Standardize',true, 'OptimizeHyperparameters','auto',...
            'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
            'expected-improvement-plus'));

% test phase: 
% using 20% of data as test set (size of K)
% creating R matrix
R=zeros(K,N);
parfor i=1:N
    R(:,i)=predict(GPRmdls{i},xTe);
end
ypred=predict(SVMmdl,R);
RMSE=mean(sqrt((yTe-ypred).^2));

end