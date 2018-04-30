% datasource = csvread('slice_localization_data.csv', 1, 0);
% randsq = randperm(size(datasource,1));
% xtr = datasource(randsq(1:100), 1:round(end/2));
% ytr = datasource(randsq(1:100), end);
% gprMdl = fitrgp(xtr, ytr, 'FitMethod','sr','BasisFunction','linear',...
%         'ActiveSetMethod','sgma','PredictMethod','fic',...
%         'KernelFunction', 'complexkernel','KernelParameters',[1 0.5 0.5]);
% gloss = zeros(1,20);
% rmse = zeros(1,20);
% 
% for i = 1:20
%     xte = datasource(randsq(11001:11000+20*i), 2:round(end/2));
%     yte = datasource(randsq(11001:11000+20*i), end);
%     gloss(i) = loss(gprMdl, xte, yte);
%     rmse(i) = sqrt(gloss(i)/i*20);
% end


datasource = csvread('slice_localization_data.csv', 1, 0);
randsq = randperm(size(datasource,1));
xtr = datasource(randsq(1:100), 2:end-1);
ytr = datasource(randsq(1:100), end);
% gprMdl = fitrgp(xtr, ytr, 'FitMethod','sr','BasisFunction','linear',...
%         'ActiveSetMethod','sgma','PredictMethod','fic',...
%         'KernelFunction', 'ardsquaredexponential','KernelParameters',[1 0.5 0.5]);
gprMdl = fitrgp(xtr,ytr,'KernelFunction', 'squaredexponential');

% xte = datasource(randsq(11001:11000+4000), 2:end-1);
% yte = datasource(randsq(11001:11000+4000), end);
% ypred = predict(gprMdl, xte);  
% rmse = sqrt(mean(yte - ypred).^2);

gloss = zeros(1,20);
rmse = zeros(1,20);

for i = 1:20
    xte = datasource(randsq(11001:11000+200*i), 2:end-1);
    yte = datasource(randsq(11001:11000+200*i), end);
    % gloss(i) = loss(gprMdl, xte, yte);
    ypred = predict(gprMdl, xte);  
    rmse(i) = sqrt(mean(yte - ypred).^2);
end

rmse = rmse';