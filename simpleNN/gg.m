%% Kaggle Competition - Digit Recognition

%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  linear exercise. You will need to complete the following functions 
%  in this exericse:
%
%     sigmoidGradient.m
%     randInitializeWeights.m
%     nnCostFunction.m
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%

%% Initialization
clear ; close all; clc
manualMode = 1;
fprintf('Loading Data ...\n');

M = csvread ('/Volumes/RamDisk/train.csv');
N = csvread('/Volumes/RamDisk/test.csv');
X = M(2:42001,2:785);
X = rotateStd(X);
y = M(2:42001,1);
y(y==0) = 10;
Xpred = N(2:28001,:);
Xpred = rotateStd(Xpred);
K = 100;

fprintf('Doing principle components analysis ...\n');
L = [X; Xpred];
%L = deepen(L, 194);
Z = redu_dim(L, K);
Zpred = Z((1 + size(X, 1)):end, :);
Z = Z(1:size(X, 1), :);

fprintf('Randomly spliting as training set, cv set and test set ...\n');
sel = randperm (size(X,1));
tempN = size(sel,2);
%	Set Index for Training Set, Cross-Validation Set and Test Set.
indTrain = sel(1:(tempN * .6));
indCVal = sel((tempN * .6 + 1):(tempN * .8));
indTest = sel((tempN * .8 + 1):end);

Xtest = X(sel((tempN * .8 + 1):end),:);
Ztest = Z(sel((tempN * .8 + 1):end),:);
ytest = y(sel((tempN * .8 + 1):end));
Xval = X(sel((tempN * .6 + 1):(tempN * .8)),:);
Zval = Z(sel((tempN * .6 + 1):(tempN * .8)),:);
yval = y(sel((tempN * .6 + 1):(tempN * .8)));
X = X(sel(1:(tempN * .6)),:);
Z = Z(sel(1:(tempN * .6)),:);
y = y(sel(1:25200));
clear M N tempN		%	keep 'L' and 'sel' for retrain useage.
fprintf('Initialization competed, go or press Ctrl+C into manual mode ...\n');
if ~exist('manualMode')
	pause;
end

fprintf('Selecting lambda ...\n');
layer_size = [K 500 500 length(unique(y))]
initial_nn_params = initializeWeight(layer_size);
%lambdaVec = [(0:0.001:0.003) (0.004:0.003:0.01) (0.03:0.02:0.1) (0.1:0.3:1)]';
%lambda = seLambda(initial_nn_params, layer_size, Z, y, Zval, yval, lambdaVec);
lambda = seLambda(initial_nn_params, layer_size, Z, y, Zval, yval);
fprintf('Lambda selection completed, continue or press Ctrl+C into manual mode ...\n');
if ~exist('manualMode')
	pause;
	close all;
end

fprintf('Training ...\n');
nn_params = train(layer_size, Z, y, lambda, initial_nn_params, 500);
test(nn_params, layer_size, Z, y, Zval, yval, Ztest, ytest, lambda);
fprintf('Training completed, go onto manual mode or press Ctrl+C to quit.\n');
if ~exist('manualMode')
	pause;
end

fprintf('Writing predicted results ...\n');
pred = predict(nn_params, layer_size, Zpred);
csvwrite('/Volumes/RamDisk/results.csv', pred);
csvwrite('/Volumes/RamDisk/nn_params.csv', nn_params);
csvwrite('/Volumes/RamDisk/selInx.csv', sel);
fprintf('Task completed.\n');






