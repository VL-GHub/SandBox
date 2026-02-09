%% RIDGE REGRESSION (RELATIVE-ERROR LOSS) ON SYNTHETIC POLYNOMIAL DATA 
% This script performs ridge regression on a synthetic dataset using
% a relative-error weighting scheme. The model is fit using closed-form
% weighted ridge regression, and cross-validation is used to select
% the optimal regularization parameter (lambda).
close all;
clear;
clc;

% For reproducibility
    rng(1);


%% 1. Generate synthetic data
% Number of data points
    num_samples = 300;

% Independent variable (column vector)
    x = linspace(-2, 2, num_samples)';

% True underlying function
    y_true = 3*x - 2*x.^2 + 0.5*x.^3;

% Add Gaussian noise
    noise_std = 1;
    y_noisy = y_true + noise_std*randn(num_samples, 1);


%% TRUE COEFFICIENTS (KNOWN FOR THIS DATASET)
% Polynomial basis is:
%   y = b1*x + b2*x^2 + b3*x^3 + ... + b15*x^15
% Only first three coefficients are nonzero
beta_true = zeros(15,1);
beta_true(1) = 3.0;
beta_true(2) = -2.0;
beta_true(3) = 0.5;


%% 2. Build polynomial features (up to poly_degree)
poly_degree = 15;
X_poly = zeros(num_samples, poly_degree);
for degree = 1:poly_degree
    X_poly(:, degree) = x.^degree;
end


%% 3. Standardize predictors (mean=0, std=1)
% Mean of each polynomial feature
    X_mean = mean(X_poly);

% Standard deviation of each feature
    X_std  = std(X_poly);

X_standardized = (X_poly - X_mean) ./ X_std;


%% 4. Split data into training, validation, and test sets
% Training Percent
    per_train = 0.6;
    num_train = round(per_train*num_samples);

% Validation Percent
    per_val = 0.2;
    num_val = round(per_val*num_samples);

X_train = X_standardized(1:num_train, :);
y_train = y_noisy(1:num_train);

X_val   = X_standardized(num_train+1:num_train+num_val, :);
y_val   = y_noisy(num_train+1:num_train+num_val);

X_test  = X_standardized(num_train+num_val+1:end, :);
y_test  = y_noisy(num_train+num_val+1:end);


%% 5. Define lambda (regularization) grid
% Lambdas must be generated from TRAIN + VALIDATION
    X_full_tmp = [X_train; X_val];
    y_full_tmp = [y_train; y_val];
    num_full = size(X_full_tmp,1);

[~, FitInfo_dummy] = lasso(X_full_tmp, y_full_tmp, 'Standardize', false);
lambda_grid = FitInfo_dummy.Lambda;
num_lambdas = length(lambda_grid);


%% 6. K-Fold Cross-Validation
num_folds = 5;

% Cross-validation partition on TRAIN + VALIDATION
    cv_partition = cvpartition(num_full, 'KFold', num_folds);

% To avoid division by zero for relative error
    tolerance = 1e-3;

% Preallocate metrics arrays
    cv_relative_error = zeros(num_lambdas, 1);
    cv_mean_squared_error = zeros(num_lambdas, 1);
    cv_mean_absolute_error = zeros(num_lambdas, 1);

for lambda_index = 1:num_lambdas
    current_lambda = lambda_grid(lambda_index);
    
    fold_rel_error = zeros(num_folds, 1);
    fold_mse = zeros(num_folds, 1);
    fold_mae = zeros(num_folds, 1);
    
    for fold = 1:num_folds
        % Get training and validation indices for this fold
            train_indices = training(cv_partition, fold);
            val_indices   = test(cv_partition, fold);
        
        % Extract fold-specific training and validation data
            X_train_fold = X_full_tmp(train_indices, :);
            y_train_fold = y_full_tmp(train_indices);

            X_val_fold = X_full_tmp(val_indices, :);
            y_val_fold = y_full_tmp(val_indices);
        
        num_train_fold = size(X_train_fold, 1);
        
        % --------------------------
        % Relative-error weighting
        % --------------------------
            safe_indices = abs(y_train_fold) > tolerance;
            weights = zeros(num_train_fold, 1);
            weights(safe_indices) = 1 ./ (y_train_fold(safe_indices).^2);
        
        % --------------------------
        % Weighted ridge regression
        % --------------------------
        % Adding intercept
            X_aug = [ones(num_train_fold, 1), X_train_fold];

        lambda_matrix = current_lambda * eye(poly_degree+1);

        % Do not penalize intercept
            lambda_matrix(1, 1) = 0;
        
        % Weighted least squares with ridge penalty
            theta_hat = ((X_aug' * (weights .* X_aug)) + lambda_matrix) \ X_aug' * (weights .* y_train_fold);
        
        % Predict on validation fold
            X_val_aug = [ones(size(X_val_fold,1),1), X_val_fold];
            y_pred_val = X_val_aug * theta_hat;
        
        % --------------------------
        % Compute validation errors
        % --------------------------
        % Relative error
            safe_val_indices = abs(y_val_fold) > tolerance;

            if any(safe_val_indices)
                rel_error = (y_pred_val(safe_val_indices) - y_val_fold(safe_val_indices)) ./ y_val_fold(safe_val_indices);
                fold_rel_error(fold) = mean(rel_error.^2);
            else
                fold_rel_error(fold) = mean((y_pred_val - y_val_fold).^2); 
            end
        
        % Standard MSE & MAE
            fold_mse(fold) = mean((y_val_fold - y_pred_val).^2);
            fold_mae(fold) = mean(abs(y_val_fold - y_pred_val));
    end
    
    % Average across folds
        cv_relative_error(lambda_index) = mean(fold_rel_error);
        cv_mean_squared_error(lambda_index) = mean(fold_mse);
        cv_mean_absolute_error(lambda_index) = mean(fold_mae);
end


%% 7. Select best lambda and fit final model on full training + validation set
% Select best lambda
    [~, best_lambda_index] = min(cv_relative_error);
    best_lambda = lambda_grid(best_lambda_index);

X_full = [X_train; X_val];
y_full = [y_train; y_val];
num_full = length(y_full);

safe_full_idx = abs(y_full) > tolerance;
weights_full = zeros(num_full,1);
weights_full(safe_full_idx) = 1 ./ (y_full(safe_full_idx).^2);

X_aug_full = [ones(num_full,1), X_full];
lambda_matrix_full = best_lambda * eye(poly_degree+1);
lambda_matrix_full(1,1) = 0;

theta_final = ((X_aug_full' * (weights_full .* X_aug_full)) + lambda_matrix_full) \ X_aug_full' * (weights_full .* y_full);
intercept_final = theta_final(1);
beta_final      = theta_final(2:end);


%% 8. Predictions and error metrics
% Training + validation predictions
    y_pred_trainval = X_full * beta_final + intercept_final;

% Test set predictions
    y_pred_test = X_test * beta_final + intercept_final;

% Training/validation metrics
    Train_L2_MSE = mean((y_full - y_pred_trainval).^2);
    Train_L1_MAE = mean(abs(y_full - y_pred_trainval));
    Train_L1_Pct = (Train_L1_MAE / mean(abs(y_full))) * 100;

% Test metrics
    Test_L2_MSE = mean((y_test - y_pred_test).^2);
    Test_L1_MAE = mean(abs(y_test - y_pred_test));
    Test_L1_Pct = (Test_L1_MAE / mean(abs(y_test))) * 100;

% Validation metrics (from CV)
    Validation_Rel_Error = cv_relative_error(best_lambda_index);
    Validation_L2_MSE   = cv_mean_squared_error(best_lambda_index);
    Validation_L1_MAE   = cv_mean_absolute_error(best_lambda_index);
    Validation_L1_Pct   = (Validation_L1_MAE / mean(abs(y_full))) * 100;


%% 9. Display results
fprintf('\n===== Ridge Regression (Relative-Error Loss) =====\n');
fprintf('Best Lambda: %.3e\n', best_lambda);
fprintf('                     |   L1 %% Error   | L1 Error (MAE)  | L2 Error (MSE)\n');
fprintf('---------------------|----------------|-----------------|----------------\n');
fprintf('Training Set         |      %6.f%%   |     %9.3f   |   %10.3f\n', ...
        Train_L1_Pct, Train_L1_MAE, Train_L2_MSE);
fprintf('Validation Set (CV)  |      %6.f%%   |     %9.3f   |   %10.3f\n', ...
        Validation_L1_Pct, Validation_L1_MAE, Validation_L2_MSE);
fprintf('Test Set             |      %6.f%%   |     %9.3f   |   %10.3f\n', ...
        Test_L1_Pct, Test_L1_MAE, Test_L2_MSE);


%% 10. COEFFICIENT RECOVERY COMPARISON (ORIGINAL BASIS)
% NOTE:
% The model is trained in standardized feature space.
% Coefficients must be converted back to the original polynomial basis
% before comparison to beta_true.
% Prediction errors are invariant to this transformation and should
% be computed in standardized space.

% Convert coefficients back to original (unstandardized) polynomial basis
beta_raw = beta_final ./ X_std';
intercept_raw = intercept_final - sum(beta_final .* (X_mean' ./ X_std'));

fprintf('\n----- True vs Estimated Coefficients (Original Basis) -----\n');
fprintf('Degree |   True Value |  Estimated Value |  Error\n');
fprintf('----------------------------------------------------------\n');
for k = 1:poly_degree
    fprintf('%6d | %12.1f | %16.2f | % .2f\n', ...
        k, beta_true(k), beta_raw(k), beta_raw(k) - beta_true(k));
end

coef_relative_error = norm(beta_raw - beta_true) / norm(beta_true);
fprintf('\nRelative L2 coefficient error (original basis): %.4e\n', coef_relative_error);


%% 11. Plot CV curve (Relative-Error Loss)
figure;
semilogx(lambda_grid, cv_relative_error, 'LineWidth', 2);
hold on;
plot(best_lambda, cv_relative_error(best_lambda_index), 'ro', 'MarkerSize', 8, 'LineWidth', 2);
grid on;
xlabel('\lambda');
ylabel('Cross-Validation Relative Error');
title('Ridge Regression CV Curve (Relative-Error Loss)');
legend('CV Relative Error','Optimal \lambda','Location','best');


%% 12. Plot predicted vs true values
figure;
scatter(x, y_noisy, 25, 'b', 'filled');
hold on;
scatter(x, [y_pred_trainval; y_pred_test], 25, 'r', 'filled');
grid on;
xlabel('x');
ylabel('y');
title('Ridge Regression: Predicted vs True');
legend('Noisy Data','Predicted Values','Location','best');
