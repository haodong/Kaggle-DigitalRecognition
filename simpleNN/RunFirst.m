%% Kaggle Competition - Digit Recognition

%  Instructions
%  ------------
% 
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
net.K = 100;
if isfield(net, 'K') && net.K~=0
	net.layers = [net.K 1000 length(unique(y))];
else
	net.layers = [size(X, 2) 40 30 20 length(unique(y))];
end

fprintf('Doing principle components analysis ...\n');
L = [X; Xpred];
%L = deepen(L, 194);
Z = redu_dim(L, net.K);
Zpred = Z((1 + size(X, 1)):end, :);
Z = Z(1:size(X, 1), :);

fprintf('Randomly spliting as training set, cv set and test set ...\n');
sel = randperm (size(X, 1));
tempN = size(sel, 2);
net.index.train = sel(1:(tempN * .6));
net.index.validation = sel((tempN * .6 + 1):(tempN * .8));
net.index.test = sel((tempN * .8 + 1):end);
Xtest = X(net.index.test, :);
ytest = y(net.index.test, :);
Ztest = Z(net.index.test, :);
Xval = X(net.index.validation, :);
yval = y(net.index.validation, :);
Zval = Z(net.index.validation, :);
X = X(net.index.train, :);
y = y(net.index.train, :);
Z = Z(net.index.train, :);
clear L M N tempN sel
fprintf('Initialization competed, go or press Ctrl+C into manual mode ...\n');
if exist('manualMode')
	pause;
end

fprintf('Selecting lambda ...\n');
net.theta = initializeWeight(net.layers);
%lambdaVec = [(0:0.001:0.003) (0.004:0.003:0.01) (0.03:0.02:0.1) (0.1:0.3:1)]';
%lambda = seLambda(initial_nn_params, layer_size, Z, y, Zval, yval, lambdaVec);
net = seLambda(net, Z, y, Zval, yval);
test(net, Z, y, Zval, yval, Ztest, ytest);
fprintf('Lambda selection completed, continue or press Ctrl+C into manual mode ...\n');
if exist('manualMode')
	pause;
	close all;
end

fprintf('Training ...\n');
net = train(net, Z, y, 400);
test(net, Z, y, Zval, yval, Ztest, ytest);
fprintf('Training completed, go onto manual mode or press Ctrl+C to quit.\n');
if exist('manualMode')
	pause;
end

fprintf('Writing predicted results ...\n');
pred = predict(net, Zpred);
csvwrite('/Volumes/RamDisk/results.csv', pred);
save '/Volumes/RamDisk/nn_params.mat' net;
fprintf('Task completed.\n');






