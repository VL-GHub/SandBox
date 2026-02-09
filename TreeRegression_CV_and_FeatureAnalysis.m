%% SCRIPT: Tree-Based Feature Ranking + Linear Regression Elbow
% Compatible with MATLAB R2022b
% ------------------------------------------------------------------------
% Requirements:
% - Statistics and Machine Learning Toolbox
% - Parallel Computing Toolbox
% ------------------------------------------------------------------------
close all;
clear;
clc;

% For reproducibility
    rng(1);


%% Section 1: Generate Synthetic Polynomial Data
fprintf('--- Section 1: Generating Synthetic Polynomial Data ---\n');
% Number of data points
    nSamples = 300;

% Number of predictors
    nFeatures = 15;

% Independent variable (column vector)
    x = linspace(-2,2,nSamples)';

% True underlying function
    yTrue = 3*x - 2*x.^2 + 0.5*x.^3;

% Add Gaussian noise
    sigma = 1;
    yFull = yTrue + sigma*randn(nSamples,1);


%% Build polynomial features
XFull = zeros(nSamples,nFeatures);
featureNames = cell(nFeatures,1);
for j = 1:nFeatures
    XFull(:,j) = x.^j;
    featureNames{j} = sprintf('x^%d',j);
end


%% Section 2: Train/Test Split
% Training Percent
    trainRatio = 0.8;
    nTrain = floor(trainRatio*nSamples);

idxPerm = randperm(nSamples);

Xtrain = XFull(idxPerm(1:nTrain),:);
ytrain = yFull(idxPerm(1:nTrain));

Xtest  = XFull(idxPerm(nTrain+1:end),:);
ytest  = yFull(idxPerm(nTrain+1:end));


%% Section 3: CV over MaxNumSplits
fprintf('--- Section 3: CV over MaxNumSplits ---\n');

maxSplitsGrid = [5 10 20 40 80 160];
numFolds = 5;
minLeafSize = 10;  % Increase for large datasets

% Start parallel pool if not running
    if isempty(gcp('nocreate'))
        parpool;
    end

% Preallocate metrics arrays
    cvMSE = zeros(numel(maxSplitsGrid),1);
    cvSE  = zeros(numel(maxSplitsGrid),1);

% K-fold partition
    cvp = cvpartition(size(Xtrain,1),'KFold',numFolds);

parfor iSplit = 1:numel(maxSplitsGrid)
    tree = fitrtree(Xtrain,ytrain, 'MaxNumSplits', maxSplitsGrid(iSplit), 'MinLeafSize', minLeafSize, 'PredictorNames', featureNames);
    
    cvTree = crossval(tree,'KFold',numFolds);
    
    % MANUAL fold-wise MSE & SE
        msePerFold = zeros(numFolds,1);
        for kFold = 1:numFolds
            idxTestFold = test(cvp,kFold);
            yPredFold = kfoldPredict(cvTree);
            yPredFold = yPredFold(idxTestFold);
            yTrueFold = ytrain(idxTestFold);

            % MSE (selection)
                msePerFold(kFold) = mean((yTrueFold - yPredFold).^2);
        end

    % MSE
        cvMSE(iSplit) = mean(msePerFold);

    % SE
        cvSE(iSplit)  = std(msePerFold)/sqrt(numFolds);
end

% 1-SE rule
    [minMSE,minIdx] = min(cvMSE);
    mseThreshold = minMSE + cvSE(minIdx);
    oneSEIdx = find(cvMSE <= mseThreshold,1,'first');
    bestSplits = maxSplitsGrid(oneSEIdx);

    fprintf('Minimum CV-MSE: %.4f at MaxNumSplits = %d\n', minMSE, maxSplitsGrid(minIdx));
    fprintf('1-SE rule selects MaxNumSplits = %d\n\n', bestSplits);

% CV curve plot
figure('Name','CV MSE vs MaxNumSplits');
errorbar(maxSplitsGrid, cvMSE, cvSE,'o-','LineWidth',1.5);
xline(bestSplits,'--r','1-SE Choice');
xlabel('MaxNumSplits'); ylabel('CV MSE'); grid on;
title('Tree Complexity Selection');


%% Section 4: Stable Feature Importance
fprintf('--- Section 4: Stable Feature Importance ---\n');

numFeatures = numel(featureNames);
impMat = zeros(numFolds,numFeatures);

parfor kFold = 1:numFolds
    idxTrainFold = training(cvp,kFold);
    treeFold = fitrtree(Xtrain(idxTrainFold,:), ytrain(idxTrainFold), 'MaxNumSplits', bestSplits, 'MinLeafSize', minLeafSize, 'PredictorNames', featureNames);
    impMat(kFold,:) = predictorImportance(treeFold);
end

meanImp = mean(impMat,1);
stdImp  = std(impMat,[],1);

[sortedImp, idxSorted] = sort(meanImp,'descend');
rankedFeatures = featureNames(idxSorted);


%% Section 5: Top Features Comparison
topK1 = min(10,numFeatures);
topK2 = min(15,numFeatures);

fprintf('Top-%d Features:\n', topK1);
for i = 1:topK1
    fprintf(' %2d. %-6s %.4f ± %.4f\n', ...
        i, rankedFeatures{i}, meanImp(idxSorted(i)), stdImp(idxSorted(i)));
end

fprintf('\nTop-%d Features:\n', topK2);
for i = 1:topK2
    fprintf(' %2d. %-6s %.4f ± %.4f\n', ...
        i, rankedFeatures{i}, meanImp(idxSorted(i)), stdImp(idxSorted(i)));
end

% Plot feature importance
figure('Name','Top Feature Importance');
subplot(1,2,1);
barh(sortedImp(1:topK1));
set(gca,'YDir','reverse','YTick',1:topK1,'YTickLabel',rankedFeatures(1:topK1));
xlabel('Mean Importance');
title(sprintf('Top-%d Features',topK1));
grid on;

subplot(1,2,2);
barh(sortedImp(1:topK2));
set(gca,'YDir','reverse','YTick',1:topK2,'YTickLabel',rankedFeatures(1:topK2));
xlabel('Mean Importance');
title(sprintf('Top-%d Features',topK2));
grid on;


%% Section 6: Train Final Tree
fprintf('Training final tree with best MaxNumSplits...\n');
finalTree = fitrtree(Xtrain,ytrain, 'MaxNumSplits', bestSplits, 'MinLeafSize', minLeafSize, 'PredictorNames', featureNames);

view(finalTree,'Mode','graph');


%% Section 7: Evaluate on Test Set
yPredTest = predict(finalTree, Xtest);
testMSE = mean((ytest - yPredTest).^2);
testRMSE = sqrt(testMSE);
testR2 = 1 - sum((ytest-yPredTest).^2)/sum((ytest-mean(ytest)).^2);

fprintf('\nTest MSE  : %.4f\n', testMSE);
fprintf('Test RMSE : %.4f\n', testRMSE);
fprintf('Test R^2  : %.4f\n\n', testR2);


%% Section 8: Linear Regression Top-K Elbow (MSE + MAE)
fprintf('--- Section 8: Linear Regression Top-K Elbow ---\n');

% Split training data into linear-regression train/validation
    valRatio = 0.20;   % validation
    nTrainLR = size(Xtrain,1);
    idxLR = randperm(nTrainLR);

nVal = floor(valRatio * nTrainLR);
idxValLR = idxLR(1:nVal);
idxTrainLR = idxLR(nVal+1:end);

XtrainLR = Xtrain(idxTrainLR,:);
ytrainLR = ytrain(idxTrainLR);

XvalLR = Xtrain(idxValLR,:);
yvalLR = ytrain(idxValLR);

Kmax = numFeatures;

% Preallocate metrics
    trainMSE = zeros(Kmax,1); 
    valMSE = zeros(Kmax,1);

trainMAE = zeros(Kmax,1);
valMAE = zeros(Kmax,1);

trainMAEpct = zeros(Kmax,1);
valMAEpct = zeros(Kmax,1);

meanAbsTrain = mean(abs(ytrainLR));
meanAbsVal   = mean(abs(yvalLR));


% % Setup parallel pool with limited cores
% maxCores = 15;  % adjust based on your system and RAM
% 
% p = gcp('nocreate'); 
% if isempty(p)
%     parpool(maxCores);
% elseif p.NumWorkers ~= maxCores
%     delete(p);
%     parpool(maxCores);
% end


parfor K = 1:Kmax
    topKfeat = rankedFeatures(1:K);

    % Select top-K predictors
        Xtrain_topK = XtrainLR(:, ismember(featureNames, topKfeat));
        Xval_topK   = XvalLR(:,   ismember(featureNames, topKfeat));

    % Fit linear regression on training set only
        mdl = fitlm(Xtrain_topK, ytrainLR);

    % Training predictions
        yPredTrain = predict(mdl, Xtrain_topK);
        trainMSE(K) = mean((ytrainLR - yPredTrain).^2);
        trainMAE(K) = mean(abs(ytrainLR - yPredTrain));
        trainMAEpct(K) = (trainMAE(K) / meanAbsTrain) * 100;

    % Validation predictions
        yPredVal = predict(mdl, Xval_topK);
        valMSE(K) = mean((yvalLR - yPredVal).^2);
        valMAE(K) = mean(abs(yvalLR - yPredVal));
        valMAEpct(K) = (valMAE(K) / meanAbsVal) * 100;
end

%% Plot MSE (absolute only)
figure('Name','Top-K Feature Elbow - MSE');
plot(1:Kmax, trainMSE,'-o','LineWidth',1.2); hold on;
plot(1:Kmax, valMSE,'-s','LineWidth',1.5);
xlabel('Top-K Features'); ylabel('MSE');
legend('Train MSE','Validation MSE','Location','best');
grid on;
title('Top-K Feature Elbow (MSE)');

%% Plot MAE (absolute & percent)
figure('Name','Top-K Feature Elbow - MAE');
plot(1:Kmax, trainMAE,'-o','LineWidth',1.2);
hold on;
plot(1:Kmax, valMAE,'-s','LineWidth',1.5);
ylabel('MAE');
xlabel('Top-K Features');
legend('Train MAE','Validation MAE','Location','best');
grid on;
title('Top-K Feature Elbow (MAE)');

figure('Name','Top-K Feature Elbow - MAE %');
plot(1:Kmax, trainMAEpct,'--o','LineWidth',1.0);
hold on;
plot(1:Kmax, valMAEpct,'--s','LineWidth',1.0);
ylabel('MAE (%)');
xlabel('Top-K Features');
legend('Train MAE %','Validation MAE %','Location','best');
grid on;
title('Top-K Feature Elbow (MAE)');

fprintf('--- Section 8 Finished ---\n');
