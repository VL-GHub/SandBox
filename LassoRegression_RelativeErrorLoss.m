%% Lasso Regression (Relative-Error Loss) on Synthetic Polynomial Data
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


%% 2. Build polynomial features
poly_degree = 15;
X_poly = zeros(num_samples, poly_degree);
for degree = 1:poly_degree
    X_poly(:, degree) = x.^degree;
end


%% 3. Standardize predictors
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


%% 5. Define lambda grid from TRAIN + VALIDATION
% Combine training + validation
    X_full_tmp = [X_train; X_val];
    y_full_tmp = [y_train; y_val];
    num_full = size(X_full_tmp,1);

[~, FitInfo_dummy] = lasso(X_full_tmp, y_full_tmp, 'Standardize', false);
lambda_grid = FitInfo_dummy.Lambda;
num_lambdas = length(lambda_grid);


%% 6. 5-Fold Cross-Validation with Relative-Error Weighting
num_folds = 5;

% Cross-validation partition on TRAIN + VALIDATION
    cv_partition = cvpartition(num_full, 'KFold', num_folds);

% avoid division by zero
    tolerance = 1e-3;

% Preallocate metrics arrays
    cv_relative_error = zeros(num_lambdas,1);
    cv_mean_squared_error = zeros(num_lambdas,1);
    cv_mean_absolute_error = zeros(num_lambdas,1);

for lambda_index = 1:num_lambdas
    current_lambda = lambda_grid(lambda_index);
    
    fold_rel_error = zeros(num_folds,1);
    fold_mse = zeros(num_folds,1);
    fold_mae = zeros(num_folds,1);
    
    for fold = 1:num_folds
        % Get training and validation indices for this fold
            train_idx = training(cv_partition, fold);
            val_idx   = test(cv_partition, fold);
        
        X_train_fold = X_full_tmp(train_idx,:);
        y_train_fold = y_full_tmp(train_idx);
        
        X_val_fold = X_full_tmp(val_idx,:);
        y_val_fold = y_full_tmp(val_idx);
        
        % --------------------------
        % Relative-error weighting
        % --------------------------
        safe_idx = abs(y_train_fold) > tolerance;
        weights = zeros(length(y_train_fold),1);
        weights(safe_idx) = 1 ./ (y_train_fold(safe_idx).^2);
    
        % Scale X and y by sqrt(weights) for weighted lasso
            X_weighted = X_train_fold .* sqrt(weights);
            y_weighted = y_train_fold .* sqrt(weights);
        
        % Fit lasso for current lambda
            [B, FitInfo] = lasso(X_weighted, y_weighted, Lambda', current_lambda, 'Standardize', false);
            beta_fold = B(:,1);
            intercept_fold = FitInfo.Intercept(1);
        
        % Predict on validation fold
            y_pred_val = X_val_fold * beta_fold + intercept_fold;
        
        % --------------------------
        % Compute validation errors
        % --------------------------
        safe_val_idx = abs(y_val_fold) > tolerance;
        if any(safe_val_idx)
            rel_error = (y_pred_val(safe_val_idx) - y_val_fold(safe_val_idx)) ./ y_val_fold(safe_val_idx);
            fold_rel_error(fold) = mean(rel_error.^2);
        else
            fold_rel_error(fold) = mean((y_pred_val - y_val_fold).^2);
        end
        
        % MSE (selection)
            fold_mse(fold) = mean((y_val_fold - y_pred_val).^2);

        % MAE (diagnostic)
            fold_mae(fold) = mean(abs(y_val_fold - y_pred_val));
    end
    
    cv_relative_error(lambda_index) = mean(fold_rel_error);

    % MSE
        cv_mean_squared_error(lambda_index) = mean(fold_mse);

    % MAE
        cv_mean_absolute_error(lambda_index) = mean(fold_mae);
end


%% 7. Select best lambda and refit on full TRAIN + VALIDATION with weighting
% Select best lambda
    [~, best_lambda_index] = min(cv_relative_error);
    best_lambda = lambda_grid(best_lambda_index);

X_full = [X_train; X_val];
y_full = [y_train; y_val];
num_full = length(y_full);

safe_full_idx = abs(y_full) > tolerance;
weights_full = zeros(num_full,1);
weights_full(safe_full_idx) = 1 ./ (y_full(safe_full_idx).^2);

X_weighted_full = X_full .* sqrt(weights_full);
y_weighted_full = y_full .* sqrt(weights_full);

[B_best, FitInfo_best] = lasso(X_weighted_full, y_weighted_full, 'Lambda', best_lambda, 'Standardize', false);
beta_final = B_best(:,1);
intercept_final = FitInfo_best.Intercept(1);


%% 8. Predictions and error metrics
% Training + validation predictions
    y_pred_trainval = X_full * beta_final + intercept_final;

% Test set predictions
    y_pred_test     = X_test * beta_final + intercept_final;

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
fprintf('\n===== Lasso Regression (Relative-Error Loss) =====\n');
fprintf('Best Lambda: %.3e\n', best_lambda);
fprintf('                     |   L1 %% Error   | L1 Error (MAE)  | L2 Error (MSE)\n');
fprintf('---------------------|----------------|-----------------|----------------\n');
fprintf('Training Set         |      %6.f%%   |     %9.3f   |   %10.3f\n', ...
        Train_L1_Pct, Train_L1_MAE, Train_L2_MSE);
fprintf('Validation Set (CV)  |      %6.f%%   |     %9.3f   |   %10.3f\n', ...
        Validation_L1_Pct, Validation_L1_MAE, Validation_L2_MSE);
fprintf('Test Set             |      %6.f%%   |     %9.3f   |   %10.3f\n', ...
        Test_L1_Pct, Test_L1_MAE, Test_L2_MSE);


%% 10. Coefficient recovery (original polynomial basis)
% NOTE:
% The model is trained in standardized feature space.
% Coefficients must be converted back to the original polynomial basis
% before comparison to beta_true.
% Prediction errors are invariant to this transformation and should
% be computed in standardized space.

% Convert coefficients back to original (unstandardized) basis
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
title('Lasso Regression CV Curve (Relative-Error Loss)');
legend('CV Relative Error','Optimal \lambda','Location','best');


%% 12. Plot predicted vs true values
figure;
scatter(x, y_noisy, 25, 'b', 'filled');
hold on;
scatter(x, [y_pred_trainval; y_pred_test], 25, 'r', 'filled');
grid on;
xlabel('x');
ylabel('y');
title('Lasso Regression: Predicted vs True');
legend('Noisy Data','Predicted Values','Location','best');
