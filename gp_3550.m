% This is a experiment for position prediction for reference within range [35, 50]
% Training:         id: 0~74
% Test id:          

% Kernel function:  Squared Exponential
% Result RMSE:      

% Kernel function:  matern32
% Result RMSE:      

% Kernel function:  matern52
% Result RMSE:      

% Kernel function:  ardsquaredexponential
% Result RMSE: 

% SVM 
% RMSE:             

datasource = csvread('slice_localization_data.csv', 1, 0);

trainingData = datasource(1:45177,:); % patient id: 0~74

% Filter data by reference in range [50,60]
id = find(trainingData(:,end) > 35 & trainingData(:,end) < 50);
training_filtered = datasource(id(:),:);

% Training sample and training label
randtr = randperm(size(training_filtered,1));
xTr = training_filtered(randtr(1:10000), 2:end - 1);
yTr = training_filtered(randtr(1:10000), end);


% Generate test data
testData = datasource(45178:end,:);
testData3550 = find(testData(:,end) > 35 & testData(:,end) < 50);
xTe = testData(testData3550, 2:end-1);
yTe = testData(testData3550, end);


gprMdl = fitrgp(xTr,yTr,'KernelFunction', 'squaredexponential');


yPred = predict(gprMdl, xTe);  
RMSE = sqrt(mean((yTe - yPred).^2));

yPred_tr = resubPredict(gprMdl);
RMSE_tr = sqrt(mean((yTr - yPred_tr).^2));

% plot
b = [yPred,yTe];
b = sortrows(b,2);
plot(b,'*')
xlabel('True reference')
ylabel('Predict reference')



% Training: SVM
SVMmdl   = fitrsvm(xTr,yTr);
yPredsvm = predict(SVMmdl, xTe);  
RMSE_svm = sqrt(mean((yTe - yPredsvm).^2));

% Training: Decision Tree
treemdl   = fitrtree(xTr,yTr);
yPredtree = predict(treemdl, xTe);  
RMSE_tree = sqrt(mean((yTe - yPredtree).^2));