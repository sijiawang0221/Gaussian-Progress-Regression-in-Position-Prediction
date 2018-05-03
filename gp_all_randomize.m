datasource = csvread('slice_localization_data.csv', 1, 0);

% This is a experiment for position prediction for all reference 
% Training:         Randomize
% Test id:          

% Kernel function:  Squared Exponential
% Result RMSE:      0.

% Kernel function:  matern32
% Result RMSE:      0.

% Kernel function:  matern52
% Result RMSE:      0.

% Kernel function:  ardsquaredexponential
% Result RMSE: 

% SVM 
% RMSE:             0.





% Randomized
randsq = randperm(size(datasource,1));
xTr = datasource(randsq(1:10000), 2:end-1);
yTr = datasource(randsq(1:10000), end);

gprMdl = fitrgp(xtr,yTr,'KernelFunction', 'squaredexponential');

xTe = datasource(randsq(11001:11000+4000), 2:end-1);
yTe = datasource(randsq(11001:11000+4000), end);
ypred = predict(gprMdl, xTe);  
rmse = sqrt(mean(yTe - ypred).^2);

rmse = rmse';
comp1 = [ypred,yTe];
comp1 = sortrows(comp1,2);
plot(comp1, '.');


% Training: SVM
SVMmdl   = fitrsvm(xTr,yTr);
yPredsvm = predict(SVMmdl, xTe);  
RMSE_svm = sqrt(mean((yTe - yPredsvm).^2));

% Training: Decision Tree
treemdl   = fitrtree(xTr,yTr);
yPredtree = predict(treemdl, xTe);  
RMSE_tree = sqrt(mean((yTe - yPredtree).^2));







