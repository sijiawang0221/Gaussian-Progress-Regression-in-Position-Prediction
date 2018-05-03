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



% not randomized
trainingdata2 = datasource(1:41472,:); %70 patient
randtr2 = randperm(size(trainingdata2,1));
xtr2 = trainingdata2(randtr2(1:10000), 2:end-1);
ytr2 = trainingdata2(randtr2(1:10000), end);

testdata2 = datasource(41473:end,:);
randte2 = randperm(size(testdata2,1));

gprMdl2 = fitrgp(xtr2,ytr2,'KernelFunction', 'squaredexponential');


xte2 = testdata2(randte2(1:4000), 2:end-1);
yte2 = testdata2(randte2(1:4000), end);
ypred2 = predict(gprMdl2, xte2); 
rmse2 = sqrt(mean((yte2 - ypred2).^2));


rmse2 = rmse2';
comp2 = [ypred2,yte2];
comp2 = sortrows(comp2,2);
plot(comp2, '.');


% Training: SVM
SVMmdl   = fitrsvm(xTr,yTr);
yPredsvm = predict(SVMmdl, xTe);  
RMSE_svm = sqrt(mean((yTe - yPredsvm).^2));








