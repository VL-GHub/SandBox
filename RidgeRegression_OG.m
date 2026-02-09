%% Ridge Regression (5-fold CV) ON SYNTHETIC POLYNOMIAL DATA
% This script performs ridge regression on a synthetic dataset
close all;
clear;
clc;

% For reproducibility
    rng(1);


%% 1. Generate synthetic data
% Number of data points
    n = 300;

% Independent variable (column vector)
    x = linspace(-2,2,n)'; 

% True underlying function
    y_true = 3*x - 2*x.^2 + 0.5*x.^3;

% Add Gaussian noise
    sigma = 1;
    y = y_true + sigma*randn(n,1);


%% TRUE COEFFICIENTS (KNOWN FOR THIS DATASET)
% Polynomial basis is:
%   y = b1*x + b2*x^2 + b3*x^3 + ... + b15*x^15
% Only first three coefficients are nonzero
beta_true = zeros(15,1);
beta_true(1) = 3.0;
beta_true(2) = -2.0;
beta_true(3) = 0.5;


%% 2. Build polynomial features (original basis X)
p = 15;  
X = zeros(n,p);
for j = 1:p
    X(:,j) = x.^j;
end


%% 3. Standardize predictors (mean=0, std=1)
% Mean of each polynomial feature
    Xmean = mean(X);

% Standard deviation of each feature
    Xstd  = std(X);

Xs = (X - Xmean) ./ Xstd;


%% 4. Split data into training, validation, test sets
% Training Percent
    per_train = 0.6;
    num_train = round(per_train*n);

% Validation Percent
    per_val = 0.2;
    num_val   = round(per_val*n);

X_train = Xs(1:num_train,:);
y_train = y(1:num_train);

X_val   = Xs(num_train+1:num_train+num_val,:);
y_val = y(num_train+1:num_train+num_val);

X_test  = Xs(num_train+num_val+1:end,:);
y_test = y(num_train+num_val+1:end);

% Combine training + validation for CV
    X_combined = [X_train; X_val];
    y_combined = [y_train; y_val];
    n_combined = size(X_combined,1);


%% 5. Generate lambda grid using lasso
[~, FitInfo_dummy] = lasso(X_combined, y_combined, 'Standardize', false);
lambdas = FitInfo_dummy.Lambda;
numL = length(lambdas);


%% 6. 5-fold cross-validation
K = 5;

% Cross-validation partition on TRAIN + VALIDATION
    cv = cvpartition(n_combined, 'KFold', K);

% Preallocate metrics arrays
    cvError = zeros(numL,1);
    cvMAE = zeros(numL,1);

for li = 1:numL
    lambda = lambdas(li);

    foldErr = zeros(K,1);
    foldMAE = zeros(K,1);

    for k = 1:K
        % Get training and validation indices for this fold
            trID  = training(cv, k);
            valID = test(cv, k);

        % Augment X with ones for intercept
            Xtr_aug = [ones(sum(trID),1), X_combined(trID,:)];

        ytr_fold = y_combined(trID);

        Xval_aug = [ones(sum(valID),1), X_combined(valID,:)];
        yval_fold = y_combined(valID);

        % Ridge penalty (do not penalize intercept)
            lambda_matrix = lambda * eye(p+1);
            lambda_matrix(1,1) = 0;

        % Ridge solution
            w_fold = (Xtr_aug' * Xtr_aug + lambda_matrix) \ (Xtr_aug' * ytr_fold);

        % Predict on validation fold
            ypred = Xval_aug * w_fold;

        % MSE (selection)
            foldErr(k) = mean((ypred - yval_fold).^2);

        % MAE (diagnostic)
            foldMAE(k) = mean(abs(ypred - yval_fold));

    end

    % MSE
        cvError(li) = mean(foldErr);

    % MAE
        cvMAE(li)   = mean(foldMAE);

end


%% 7. Select best lambda and fit final model on full training + validation set
% Select best lambda
    [bestErr, bestIdx] = min(cvError);
    bestLambda = lambdas(bestIdx);

X_aug = [ones(n_combined,1), X_combined];
lambda_matrix_full = bestLambda * eye(p+1);
lambda_matrix_full(1,1) = 0;

w_best = (X_aug' * X_aug + lambda_matrix_full) \ (X_aug' * y_combined);
intercept_final = w_best(1);
beta_final      = w_best(2:end);


%% 8. Predictions and error metrics
% Training + validation predictions
    y_pred_trainval = X_combined * beta_final + intercept_final;

% Test set predictions
    y_pred_test = X_test * beta_final + intercept_final;

% Training/validation metrics
    Train_L2_MSE = mean((y_combined - y_pred_trainval).^2);
    Train_L1_MAE = mean(abs(y_combined - y_pred_trainval));
    Train_L1_Pct = (Train_L1_MAE / mean(abs(y_combined))) * 100;

% Test metrics
    Test_L2_MSE = mean((y_test - y_pred_test).^2);
    Test_L1_MAE = mean(abs(y_test - y_pred_test));
    Test_L1_Pct = (Test_L1_MAE / mean(abs(y_test))) * 100;

% Validation metrics (from CV)
    Validation_L2_MSE = bestErr;
    Validation_L1_MAE = cvMAE(bestIdx);
    Validation_L1_Pct = (Validation_L1_MAE / mean(abs(y_combined))) * 100;


%% 9. Display results
fprintf('\n===== Ridge Regression (Traditional) =====\n');
fprintf('Best lambda: %.3e\n', bestLambda);
fprintf('                     | L1 %% Error | L1 Error (MAE) | L2 Error (MSE)\n');
fprintf('---------------------|------------|----------------|----------------\n');
fprintf('Training Set         |   %6.f%%  |   %9.3f    |   %10.3f\n', ...
        Train_L1_Pct, Train_L1_MAE, Train_L2_MSE);
fprintf('Validation Set (CV)  |   %6.f%%  |   %9.3f    |   %10.3f\n', ...
        Validation_L1_Pct, Validation_L1_MAE, Validation_L2_MSE);
fprintf('Test Set             |   %6.f%%  |   %9.3f    |   %10.3f\n', ...
        Test_L1_Pct, Test_L1_MAE, Test_L2_MSE);


%% 10. Coefficient recovery (original polynomial basis)
% NOTE:
% The model is trained in standardized feature space.
% Coefficients must be converted back to the original polynomial basis
% before comparison to beta_true.
% Prediction errors are invariant to this transformation and should
% be computed in standardized space.

% Convert coefficients back to original (unstandardized) basis
beta_raw = beta_final ./ Xstd';
intercept_raw = intercept_final - sum(beta_final .* (Xmean' ./ Xstd'));

fprintf('\n----- True vs Estimated Coefficients (Original Basis) -----\n');
fprintf('Degree |   True Value |  Estimated Value |  Error\n');
fprintf('----------------------------------------------------------\n');
for k = 1:p
    fprintf('%6d | %12.1f | %16.2f | % .2f\n', ...
        k, beta_true(k), beta_raw(k), beta_raw(k) - beta_true(k));
end

coef_relative_error = norm(beta_raw - beta_true) / norm(beta_true);
fprintf('\nRelative L2 coefficient error (original basis): %.4e\n', coef_relative_error);


%% 11. Plot CV curve
figure;
semilogx(lambdas, cvError, 'LineWidth', 2);
hold on;
plot(bestLambda, bestErr, 'ro', 'MarkerSize', 8, 'LineWidth', 2);
grid on;
xlabel('\lambda');
ylabel('Cross-Validation MSE');
title('Ridge Regression CV Curve');
legend('CV Error','Optimal \lambda','Location','best');


%% 12. Plot predictions vs data
figure;
scatter(x, y, 25, 'b', 'filled'); hold on;
scatter(x, [y_pred_trainval; y_pred_test], 25, 'r', 'filled');
xlabel('x'); ylabel('y');
title('Ridge Regression: Predicted vs True');
legend('Noisy Data','Predicted','Location','best');
grid on;
