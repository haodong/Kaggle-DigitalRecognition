clear ; close all; clc
fprintf('Loading Data ...\n');
load('/Volumes/RamDisk/nn_params.mat');
M = csvread ('/Volumes/RamDisk/train.csv', 1, 0);
N = csvread('/Volumes/RamDisk/test.csv', 1, 0);
X = M(:,2:785);
X = rotateStd(X);
y = M(:,1);
y(y==0) = 10;
Xpred = rotateStd(N);

fprintf('Doing principle components analysis ...\n');
L = [X; Xpred];
%L = deepen(L, 194);
Z = redu_dim(L, net.K);
Zpred = Z((1 + size(X, 1)):end, :);
Z = Z(1:size(X, 1), :);

fprintf('Randomly spliting as training set, cv set and test set ...\n');
Xtest = X(net.index.test, :);
ytest = y(net.index.test, :);
Ztest = Z(net.index.test, :);
Xval = X(net.index.validation, :);
yval = y(net.index.validation, :);
Zval = Z(net.index.validation, :);
X = X(net.index.train, :);
y = y(net.index.train, :);
Z = Z(net.index.train, :);
clear L M N
fprintf('Initialization competed, go or press Ctrl+C into manual mode ...\n');

fprintf('Training ...\n');
test(net, Z, y, Zval, yval, Ztest, ytest);
net = train(net, Z, y, 1000);
test(net, Z, y, Zval, yval, Ztest, ytest);
fprintf('Training completed.\n');

fprintf('Writing predicted results ...\n');
pred = predict(net, Zpred);
csvwrite('/Volumes/RamDisk/results.csv', pred);
save '/Volumes/RamDisk/nn_params.mat' net;
fprintf('Task completed.\n');
