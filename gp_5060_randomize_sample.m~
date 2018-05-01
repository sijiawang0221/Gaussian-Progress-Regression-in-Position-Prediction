

% This is a experiment for position prediction for reference within range [50,60]
% Training:         id: 0 ~ 70
% Test id:          71 ~ 96

% Kernel function:  Squared Exponential
% Result RMSE:      0.2231

% Kernel function:  matern32
% Result RMSE:      0.2421

% Kernel function:  matern52
% Result RMSE:      0.2385

% Kernel function:  ardsquaredexponential
% Result RMSE:      


% Filter data by reference in range [50,60]
datasource = csvread('slice_localization_data.csv', 1, 0);
id = find(datasource(:,end) > 50 & datasource(:,end) < 60);
dataFiltered = datasource(id(:),:);

% Randomize dataset
rand = randperm(size(dataFiltered,1));
dataFiltered = dataFiltered(rand(:),:);

% Training sample and training label
xTr = dataFiltered(1:5000, 2:end - 1);
yTr = dataFiltered(1:5000, end);

% Generate test data
xTe = dataFiltered(5001:end, 2:end-1);
yTe = dataFiltered(5001:end, end);

% Training
gprMdl = fitrgp(xTr,yTr,'KernelFunction', 'ardsquaredexponential');

% Calculate RMSE
yPred = predict(gprMdl, xTe);  
RMSE = sqrt(mean((yTe - yPred).^2));















