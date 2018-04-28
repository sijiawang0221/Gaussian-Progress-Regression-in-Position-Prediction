% parameters
n            = 39450; % number of training points
input_scale  = 1;   % input scale ?
output_scale = 1;   % output scale ?
noise        = 0.5; % noise scale ?

% training locations
x = X_train;

% define GP model

% prior mean ?(x) = 0
mean_function = {@meanZero};

% prior covariance
% K(x, x'; ?, ?) = ?² exp(-|x - x'|² / 2?²)
covariance_function = {@covSEiso};

% hyperparameters
hyperparameters.mean = [];

% note: parameterization is [log(?); log(?)]
hyperparameters.cov  = ...
    [log(input_scale); ...
     log(output_scale)];

% note: parameterization is log(?))
hyperparameters.lik = log(noise);

% prior p(y | X, ?)

% mean of f(X), ?(X)
mu = feval(mean_function{:}, hyperparameters.mean, x);

% covariance of f(X), K(X, X)
K_f = feval(covariance_function{:}, hyperparameters.cov, x);

% covariance of y(X), K(X, X) + ?² I
K_y = K_f + noise^2 * eye(n);

% sample from GP prior
y = Y_train;

figure(1);
hold('off');
plot(x, y, '.');

% can we learn the right function?

% negative log marginal likelihood log p(y | X, ?)
% call gp() in "training mode")

training_nml = gp(hyperparameters, [], mean_function, covariance_function, ...
                  [], x, y);

% learn hyperparameters by gradient descent

learned_hyperparameters = minimize(hyperparameters, @gp, 100, [], ...
        mean_function, covariance_function, [], x, y);

% make predictions
% call gp() in "prediction mode"

% test locations
x_star = X_test;

[y_mean, y_var] = gp(learned_hyperparameters, [], mean_function, ...
        covariance_function, [], x, y, x_star);

hold('on');
plot(x_star, y_mean, 'r');
plot(x_star, y_mean + 2 * sqrt(y_var), 'r');
plot(x_star, y_mean - 2 * sqrt(y_var), 'r');

% try sparse GP with same data

% inducing point locations x?
x_bar = X_val;

% wrapper for covariance
sparse_covariance_function = {@apxSparse, covariance_function, x_bar};

% wrapper for inference method (computes marginal likelihoods)
inference_method = @(varargin) infGaussLik(varargin{:}, struct('s', 1));

% sparse predictions
[y_mean, y_var] = gp(learned_hyperparameters, inference_method, ...
        mean_function, sparse_covariance_function, @likGauss, x, y, x_star);

figure(2);
hold('off');
plot(x, y, '.');
hold('on');
plot(x_star, y_mean, 'r');
plot(x_star, y_mean + 2 * sqrt(y_var), 'r');
plot(x_star, y_mean - 2 * sqrt(y_var), 'r');
plot(x_bar, -2.5, 'b+');

hyperparameters.xu = x_bar;

% calling minimize next will learn inducing point locations!

sparse_learned_hyperparameters = minimize(hyperparameters, @gp, 100, ...
        inference_method, mean_function, sparse_covariance_function, ...
        [], x, y);

figure(3);
hold('off');
plot(x, y, '.');
hold('on');
plot(x_star, y_mean, 'r');
plot(x_star, y_mean + 2 * sqrt(y_var), 'r');
plot(x_star, y_mean - 2 * sqrt(y_var), 'r');
plot(sparse_learned_hyperparameters.xu, -2.5, 'b+');