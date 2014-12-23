fprintf('Training ...\n');
net = train(net, Z, y, 1000);
test(net, Z, y, Zval, yval, Ztest, ytest);
fprintf('Training completed.\n');

fprintf('Writing predicted results ...\n');
pred = predict(net, Zpred);
csvwrite('/Volumes/RamDisk/results.csv', pred);
save '/Volumes/RamDisk/nn_params.mat' net;
fprintf('Task completed.\n');
