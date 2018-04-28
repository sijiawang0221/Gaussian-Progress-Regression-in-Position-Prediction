datasource = csvread('slice_localization_data.csv', 1, 0);
randsq = randperm(size(datasource,1));
xtr = datasource(randsq(1:100), 2:round(end/2));
ytr = datasource(randsq(1:100), end);
gprMdl = fitrgp(xtr, ytr, 'FitMethod','sr','BasisFunction','linear',...
        'ActiveSetMethod','sgma','PredictMethod','fic',...
        'KernelFunction', 'complexkernel','KernelParameters',[1 0.5 0.5]);
gloss = zeros(1,20);
rmse = zeros(1,20);

for i = 1:20
    xte = datasource(randsq(11001:11000+20*i), 2:round(end/2));
    yte = datasource(randsq(11001:11000+20*i), end);
    gloss(i) = loss(gprMdl, xte, yte);
    rmse(i) = sqrt(gloss(i)/i*20);
end