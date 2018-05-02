

% This is a experiment for position prediction for reference within range [50,60]
% Training:         id: 0 ~ 70
% Test id:          71 ~ 96

% Kernel function:  Squared Exponential
% Result RMSE:      1.9926

% Kernel function:  matern32
% Result RMSE:      2.0079

% Kernel function:  matern52
% Result RMSE:      1.9996

% Kernel function:  ardsquaredexponential
% Result RMSE:      

datasource = csvread('slice_localization_data.csv', 1, 0);

trainingData = datasource(1:40177,:); % patient id: 0~70

% Filter data by reference in range [50,60]
trainingData5060 = find(trainingData(:,end) > 50 & trainingData(:,end) < 60);

% Training sample and training label
xTr = datasource(trainingData5060, 2:end - 1);
yTr = datasource(trainingData5060, end);

% hyperparameter
theta = [mean(std(xTr));std(yTr)/sqrt(2)];

% Generate test data
testData = datasource(40178:end,:);
testData5060 = find(testData(:,end) > 50 & testData(:,end) < 60);
xTe = testData(testData5060, 2:end-1);
yTe = testData(testData5060, end);

% Training: GPR
gprMdl = fitrgp(xTr,yTr,'PredictMethod','sr','FitMethod', 'fic','KernelFunction', 'squaredexponential','Sigma', 20);

% Calculate RMSE
yPred = predict(gprMdl, xTe);  
RMSE = sqrt(mean((yTe - yPred).^2));

yPred_tr = predict(gprMdl, xTr);
dif = yTr-yPred_tr;
plot(dif);

% Training: SVM
SVMmdl = fitrsvm(xTr,yTr);
% ,'KernelFunction','gaussian','KernelScale',...
%             'auto','Standardize',true
%         , 'OptimizeHyperparameters','auto',...
%             'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
%             'expected-improvement-plus'));

yPredsvm = predict(SVMmdl, xTe);  
RMSE = sqrt(mean((yTe - yPredsvm).^2));

yPred_tr = predict(SVMmdl, xTr);
dif = yTr-yPred_tr;











