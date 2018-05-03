% Final version

% This is a experiment for position prediction for reference within range [35, 50]
% Training:         Randomize
% Test id:          

% Kernel function:  Squared Exponential
% Result RMSE:      0.1575

% Kernel function:  matern32
% Result RMSE:      0.1574

% Kernel function:  matern52
% Result RMSE:      0.1550

% Kernel function:  ardsquaredexponential
% Result RMSE: 

% SVM 
% RMSE:             1.2598


% Filter data by reference in range [50,60]
datasource = csvread('slice_localization_data.csv', 1, 0);
id = find(datasource(:,end) > 35 & datasource(:,end) < 50);
dataFiltered = datasource(id(:),:);

% Randomize dataset
rand = randperm(size(dataFiltered,1));
dataFiltered = dataFiltered(rand(:),:);

% Training sample and training label
xTr = dataFiltered(1:10000, 2:end - 1);
yTr = dataFiltered(1:10000, end);

% Generate test data
xTe = dataFiltered(10001:end, 2:end-1);
yTe = dataFiltered(10001:end, end);

% Training: GPR
gprMdl = fitrgp(xTr,yTr,'KernelFunction', 'squaredexponential');

% Calculate RMSE
yPred = predict(gprMdl, xTe);  
RMSE  = sqrt(mean((yTe - yPred).^2));






% Training: SVM
SVMmdl   = fitrsvm(xTr,yTr);
yPredsvm = predict(SVMmdl, xTe);  
RMSE_svm = sqrt(mean((yTe - yPredsvm).^2));

% Training: Decision Tree
treemdl   = fitrtree(xTr,yTr);
yPredtree = predict(treemdl, xTe);  
RMSE_tree = sqrt(mean((yTe - yPredtree).^2));













